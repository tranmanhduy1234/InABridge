"""InstructBLIP-style stage 2 with DINOv3, instruction-aware Q-Former and Qwen3."""

import json
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.languagemodel import QwenLanguageModel


class ModelStage2(nn.Module):
    def __init__(self, stage1, language_model, qformer_tokenizer, train_qformer=True,
                 image_size=None, normalization=((.485, .456, .406), (.229, .224, .225)),
                 interpolation='bicubic', antialias=True, qformer_max_length=128):
        super().__init__()
        projector = language_model.projector
        if projector is None or projector.fc1.in_features != stage1.qformer_model.hidden_dim:
            raise ValueError('Language model projector must match the Q-Former hidden dimension')
        self.language_model = language_model
        self.vision_encoder = stage1.vision_encoder.to(language_model.device).requires_grad_(False)
        self.qformer = stage1.qformer_model.to(language_model.device).requires_grad_(train_qformer)
        self.embeddings = stage1.embeddings.to(language_model.device).requires_grad_(False)
        self.qformer_tokenizer = qformer_tokenizer
        self.train_qformer = train_qformer
        self.preprocess_config = dict(
            image_size=image_size or self.vision_encoder.model.config.image_size, normalization=normalization,
            interpolation=interpolation, antialias=antialias,
            qformer_max_length=min(qformer_max_length, self.qformer.cfg.max_position_embeddings))
        size = self.preprocess_config['image_size']
        steps = [transforms.Resize((size, size), interpolation=InterpolationMode(interpolation), antialias=antialias),
                 transforms.ToTensor()]
        if normalization is not None:
            steps.append(transforms.Normalize(*normalization))
        self.image_transform = transforms.Compose(steps)
        self.train()

    @property
    def tokenizer(self):
        return self.language_model.tokenizer

    @property
    def projector(self):
        return self.language_model.projector

    @classmethod
    def from_stage1(cls, checkpoint, *, train_qformer=True, adapter_path=None, **llm_options):
        from src.trainingph1.engine_checkpoint import load_pretrained

        stage1, tokenizer, settings = load_pretrained(checkpoint)
        llm = QwenLanguageModel.from_pretrained(
            qformer_dim=stage1.qformer_model.hidden_dim, adapter_path=adapter_path, **llm_options)
        preprocessing = {name: settings[key] for name, key in (
            ('image_size', 'IMAGE_SIZE'), ('normalization', 'NORMALIZATION'),
            ('interpolation', 'INTERPOLATION'), ('antialias', 'ANTIALIAS'),
            ('qformer_max_length', 'MAX_LENGTH')) if key in settings}
        if adapter_path and (Path(adapter_path) / 'stage2.json').exists():
            preprocessing = json.loads((Path(adapter_path) / 'stage2.json').read_text())
        model = cls(stage1, llm, tokenizer, train_qformer, **preprocessing)
        if adapter_path and (Path(adapter_path) / 'qformer.pt').exists():
            model.qformer.load_state_dict(torch.load(Path(adapter_path) / 'qformer.pt',
                                                    map_location=llm.device, weights_only=True))
        if not llm_options.get('trainable', True):
            model.requires_grad_(False).eval()
        return model

    def train(self, mode=True):
        super().train(mode)
        self.vision_encoder.eval()
        self.embeddings.eval()
        self.qformer.train(mode and self.train_qformer)
        return self

    def encode_image(self, images, qformer_input_ids, qformer_attention_mask):
        parameter = next(self.vision_encoder.parameters())
        features = self.vision_encoder(images.to(device=parameter.device, dtype=parameter.dtype))
        tokens = qformer_input_ids.to(parameter.device)
        mask = qformer_attention_mask.to(parameter.device).bool()
        # ITM's bidirectional query/text mask lets the instruction influence visual queries.
        return self.qformer(features, self.embeddings(input_ids=tokens), mask, 'itm')['query_output']

    def forward(self, images, qformer_input_ids, qformer_attention_mask,
                input_ids, attention_mask=None, labels=None, **kwargs):
        return self.language_model(input_ids, attention_mask, labels,
                                   query_features=self.encode_image(images, qformer_input_ids, qformer_attention_mask),
                                   **kwargs)

    @torch.no_grad()
    def generate(self, images, qformer_input_ids, qformer_attention_mask, input_ids, attention_mask=None, **kwargs):
        training = self.training
        self.eval()
        try:
            return self.language_model.generate(input_ids, attention_mask,
                                                query_features=self.encode_image(images, qformer_input_ids,
                                                                                 qformer_attention_mask), **kwargs)
        finally:
            self.train(training)

    @torch.no_grad()
    def generation(self, image, prompt, *, enable_thinking=False, **kwargs):
        """Answer one PIL image (or local image path) and one text prompt."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must be nonempty text')
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert('RGB')
        if not isinstance(image, Image.Image):
            raise TypeError('image must be a PIL image or local image path')
        images = self.image_transform(image.convert('RGB')).unsqueeze(0)
        instruction = self.qformer_tokenizer(
            prompt, return_tensors='pt', truncation=True, max_length=self.preprocess_config['qformer_max_length'])
        chat = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking)
        text = self.tokenizer(chat, add_special_tokens=False, return_tensors='pt')
        output = self.generate(images, instruction['input_ids'], instruction['attention_mask'],
                               text['input_ids'], text['attention_mask'], **kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def save_pretrained(self, path):
        """Export stage 2 weights; reuse the original stage 1 checkpoint for frozen vision."""
        self.language_model.save_pretrained(path)
        torch.save(self.qformer.state_dict(), Path(path) / 'qformer.pt')
        (Path(path) / 'stage2.json').write_text(json.dumps(self.preprocess_config, indent=2) + '\n')
