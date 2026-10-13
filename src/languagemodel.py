"""Qwen3 with LoRA/QLoRA and optional Q-Former visual prefix."""

import json
import os
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.projector import QwenProjector


def prepare_lora(model, *, rank=16, alpha=32, dropout=.05, gradient_checkpointing=True,
                 adapter_path=None, trainable=True):
    model.requires_grad_(False)
    if trainable:
        if getattr(model, 'is_loaded_in_4bit', False):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
    if adapter_path:
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    if not trainable:
        return model
    return get_peft_model(model, LoraConfig(
        task_type='CAUSAL_LM', r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules='all-linear', bias='none'))

class QwenLanguageModel(nn.Module):
    def __init__(self, model, tokenizer, qformer_dim=None, projector_hidden_dim=None, dropout=0.):
        super().__init__()
        if model.config.model_type != 'qwen3':
            raise ValueError('Expected a Qwen3 causal language model')
        self.model, self.tokenizer = model, tokenizer
        hidden = projector_hidden_dim or model.config.hidden_size
        self.projector = (QwenProjector(qformer_dim, model.config.hidden_size, hidden, dropout)
                          if qformer_dim is not None else None)
        if self.projector is not None:
            self.projector.to(device=self.device, dtype=torch.float32)
        self.projector_config = dict(qformer_dim=qformer_dim, projector_hidden_dim=hidden, dropout=dropout)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @classmethod
    def from_pretrained(cls, model_id='Qwen/Qwen3-4B', *, adapter_path=None, load_in_4bit=True,
                        device=None, dtype=None, trainable=True, qformer_dim=None,
                        projector_hidden_dim=None, projector_dropout=0., rank=16, alpha=32,
                        lora_dropout=.05, gradient_checkpointing=True, **load_kwargs):
        device = torch.device(device or f'cuda:{os.environ.get("LOCAL_RANK", "0")}')
        if load_in_4bit and (device.type != 'cuda' or not torch.cuda.is_available()):
            raise ValueError('QLoRA requires CUDA; use load_in_4bit=False, device="cpu" for CPU LoRA')
        dtype = dtype or (torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported()
                          else torch.float32)
        if adapter_path:
            model_id = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
        options = dict(device_map={'': str(device)}, dtype=dtype, **load_kwargs)
        if load_in_4bit:
            options['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(model_id, **options)
        tokenizer_source = adapter_path if adapter_path and (Path(adapter_path) / 'tokenizer_config.json').exists() else model_id
        tokenizer_options = {key: load_kwargs[key] for key in ('revision', 'cache_dir', 'local_files_only', 'token')
                             if key in load_kwargs}
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_options)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError('Tokenizer needs a PAD or EOS token')
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = prepare_lora(model, rank=rank, alpha=alpha, dropout=lora_dropout,
                             gradient_checkpointing=gradient_checkpointing, adapter_path=adapter_path,
                             trainable=trainable)
        projector = dict(qformer_dim=qformer_dim, projector_hidden_dim=projector_hidden_dim,
                         dropout=projector_dropout)
        metadata = Path(adapter_path) / 'projector.json' if adapter_path else None
        if metadata and metadata.exists():
            projector = json.loads(metadata.read_text())
        result = cls(model, tokenizer, **projector)
        if metadata and metadata.exists() and result.projector is not None:
            result.projector.load_state_dict(torch.load(Path(adapter_path) / 'projector.pt',
                                                       map_location=result.device, weights_only=True))
        if not trainable:
            result.requires_grad_(False).eval()
        return result

    def _inputs(self, input_ids, attention_mask=None, query_features=None, labels=None):
        if input_ids.ndim != 2 or not input_ids.numel():
            raise ValueError('input_ids must be a nonempty [batch, length] tensor')
        ids = input_ids.to(self.device)
        mask = torch.ones_like(ids, dtype=torch.bool) if attention_mask is None else attention_mask.to(self.device).bool()
        if mask.shape != ids.shape or not mask.any(-1).all():
            raise ValueError('attention_mask must match input_ids and contain text for every sample')
        embeddings = self.model.get_input_embeddings()(ids)
        if labels is not None:
            if labels.shape != ids.shape:
                raise ValueError('labels must match input_ids')
            labels = labels.to(self.device).masked_fill(~mask, -100)
        if query_features is not None:
            if self.projector is None:
                raise ValueError('Set qformer_dim to enable the visual projector')
            if (query_features.ndim != 3 or query_features.shape[0] != ids.shape[0]
                    or query_features.shape[1] == 0 or query_features.shape[2] != self.projector.fc1.in_features):
                raise ValueError('query_features must have shape [batch, queries, qformer_dim]')
            parameter = self.projector.fc1.weight
            visual = self.projector(query_features.to(device=parameter.device, dtype=parameter.dtype)).to(embeddings.dtype)
            embeddings = torch.cat((visual, embeddings), dim=1)
            prefix_mask = mask.new_ones(visual.shape[:2])
            mask = torch.cat((prefix_mask, mask), dim=1)
            if labels is not None:
                labels = torch.cat((labels.new_full(visual.shape[:2], -100), labels), dim=1)
        if embeddings.shape[1] > self.model.config.max_position_embeddings:
            raise ValueError('Text plus visual prefix exceeds the model context length')
        if labels is not None and not labels[:, 1:].ne(-100).any():
            raise ValueError('Batch has no supervised next-token targets')
        return dict(inputs_embeds=embeddings, attention_mask=mask, labels=labels)

    def forward(self, input_ids, attention_mask=None, labels=None, query_features=None, **kwargs):
        inputs = self._inputs(input_ids, attention_mask, query_features, labels)
        positions = inputs['attention_mask'].long().cumsum(-1).sub(1).clamp_min(0)
        return self.model(**inputs, position_ids=positions, use_cache=False, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, query_features=None, **kwargs):
        """Return only generated token IDs; batched prompts must use left padding."""
        if attention_mask is not None and not attention_mask[:, -1].all():
            raise ValueError('Generation requires left-padded prompts')
        training = self.training
        self.eval()
        try:
            inputs = self._inputs(input_ids, attention_mask, query_features)
            inputs.pop('labels')
            options = dict(max_new_tokens=256, pad_token_id=self.tokenizer.pad_token_id,
                           eos_token_id=self.tokenizer.eos_token_id, use_cache=True) | kwargs
            return self.model.generate(**inputs, **options)
        finally:
            self.train(training)

    def save_pretrained(self, path):
        """Export adapter/projector weights; optimizer/RNG belong to the training checkpoint."""
        if not isinstance(self.model, PeftModel):
            raise ValueError('Saving an adapter requires a PEFT model')
        path = Path(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / 'projector.json').write_text(json.dumps(self.projector_config, indent=2) + '\n')
        if self.projector is not None:
            torch.save(self.projector.state_dict(), path / 'projector.pt')
