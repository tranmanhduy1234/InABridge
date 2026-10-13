"""Backend-independent stage 1 metadata and offline reconstruction."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from src import config

ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = 1

def _architecture(model):
    def model_config(cfg):
        return {k: v for k, v in cfg.to_dict().items() if k not in {'transformers_version', '_name_or_path'}}

    return {
        'args': dict(bert_name=model.bert_name, model_vision_id=model.vision_encoder.model_id,
                     return_layer=model.vision_encoder.return_layer, num_queries=model.qformer_model.num_queries,
                     cross_attn_every=model.qformer_model.cross_attn_every, hidden_dim=model.qformer_model.hidden_dim,
                     itc_dim=model.itc_encoder.query_proj.out_features, normalize_eps=model.itc_encoder.normalize_eps,
                     vision_mock=model.vision_encoder.mock),
        'bert_config': model_config(model.qformer_model.cfg),
        'vision_config': model_config(model.vision_encoder.model.config),
    }


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_metadata(model, tokenizer, settings=None, parent_checkpoint=None):
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer is required for offline reconstruction')
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend['padding'] = backend['truncation'] = None
    return {
        'format_version': FORMAT_VERSION,
        'run_id': uuid4().hex,
        'architecture': _architecture(model),
        'tokenizer': dict(backend=backend, special_tokens=tokenizer.special_tokens_map,
                          padding_side=tokenizer.padding_side, truncation_side=tokenizer.truncation_side,
                          model_max_length=tokenizer.model_max_length, model_input_names=tokenizer.model_input_names,
                          clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
                          split_special_tokens=tokenizer.split_special_tokens),
        'trainable': [name for name, p in model.named_parameters() if p.requires_grad],
        'config': {k: v for k, v in (vars(config) | (settings or {})).items() if k.isupper()},
        'parent_checkpoint': str(parent_checkpoint) if parent_checkpoint is not None else None,
    }


def build_model_tokenizer(metadata):
    """Reconstruct architecture and tokenizer offline, without loading weights."""
    from tokenizers import AddedToken, Tokenizer
    from transformers import BertConfig, DINOv3ViTConfig, PreTrainedTokenizerFast
    from src.trainingph1.model import ModelStage1

    architecture = metadata['architecture']
    model = ModelStage1(**architecture['args'], bert_config=BertConfig.from_dict(architecture['bert_config']),
                        vision_config=DINOv3ViTConfig.from_dict(architecture['vision_config']))
    trainable = set(metadata['trainable'])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)
    token_config = metadata['tokenizer']
    added = {token['content']: AddedToken(**{k: v for k, v in token.items() if k != 'id'})
             for token in token_config['backend']['added_tokens']}
    specials = {name: [added.get(t, t) for t in value] if isinstance(value, list)
                else added.get(value, value) for name, value in token_config['special_tokens'].items()}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(token_config['backend'])), **specials,
        **{k: v for k, v in token_config.items() if k not in {'backend', 'special_tokens'}},
    )
    return model, tokenizer


