import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from src import config


ROOT = Path(__file__).resolve().parents[2]
MANAGER_PATH = ROOT / "src/checkpoints/stage1/manager.json"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _manager():
    manager = json.loads(MANAGER_PATH.read_text())
    if manager["schema_version"] != 3:
        raise ValueError("Unsupported checkpoint manager schema")
    fields = [name for group in ("compatibility_fields", "resume_fields", "runtime_fields", "demo_fields")
              for name in manager[group]]
    configured = {name for name in vars(config) if name.isupper()}
    if len(fields) != len(set(fields)) or set(fields) != configured:
        raise ValueError(f"Checkpoint config classification mismatch: {sorted(set(fields) ^ configured)}")
    return manager


def _compatibility(model, tokenizer, settings=None):
    manager = _manager()
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if model_class != manager["model_class"]:
        raise ValueError(f"Expected {manager['model_class']}, got {model_class}")
    if not tokenizer.is_fast:
        raise ValueError("Checkpoint compatibility requires a fast tokenizer")
    if settings is None:
        settings = getattr(model, "checkpoint_config", vars(config))
    settings = {name: settings[name] for name in manager["compatibility_fields"]}
    # Architecture metadata must describe the actual model, even if global config changed.
    settings.update(
        BERT_MODEL_ID=model.bert_name,
        VISION_MODEL_ID=model.vision_encoder.model_id,
        VISION_RETURN_LAYER=model.vision_encoder.return_layer,
        NUM_QUERIES=model.qformer_model.num_queries,
        CROSS_ATTN_EVERY=model.qformer_model.cross_attn_every,
        HIDDEN_DIM=model.qformer_model.hidden_dim,
        ITC_DIM=model.itc_encoder.query_proj.out_features,
        NORMALIZE_EPS=model.itc_encoder.normalize_eps,
    )
    if not settings["TOKENIZER_USE_FAST"]:
        raise ValueError("TOKENIZER_USE_FAST must be True for checkpoints")
    if max(tokenizer.get_vocab().values()) >= model.embeddings.word_embeddings.num_embeddings:
        raise ValueError("Tokenizer token IDs exceed the model vocabulary")
    if not 0 < settings["MAX_LENGTH"] <= model.qformer_model.cfg.max_position_embeddings:
        raise ValueError("MAX_LENGTH exceeds the BERT position embedding limit")
    model.vision_encoder.compute_grid_shape(settings["IMAGE_SIZE"], settings["IMAGE_SIZE"])
    if not -model.vision_encoder.model.config.num_hidden_layers - 1 <= settings["VISION_RETURN_LAYER"] <= model.vision_encoder.model.config.num_hidden_layers:
        raise ValueError("VISION_RETURN_LAYER is outside the vision hidden states")
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # These are per-call batching options, not persistent tokenizer semantics.
    backend["padding"] = backend["truncation"] = None
    return {
        "manager": manager,
        "config": settings,
        "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in manager["source_files"]},
        "versions": {name: version(name) for name in ("torch", "transformers", "tokenizers", "torchvision")},
        "vision_config": model.vision_encoder.model.config.to_dict(),
        "vision_model_id": model.vision_encoder.model_id,
        "return_layer": model.vision_encoder.return_layer,
        "qformer_config": model.qformer_model.cfg.to_dict(),
        "normalize_eps": model.itc_encoder.normalize_eps,
        "tokenizer": {
            "backend": backend,
            "special_tokens": tokenizer.special_tokens_map,
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "model_max_length": tokenizer.model_max_length,
            "model_input_names": tokenizer.model_input_names,
            "clean_up_tokenization_spaces": tokenizer.clean_up_tokenization_spaces,
            "split_special_tokens": tokenizer.split_special_tokens,
        },
        "parameters": {name: list(value.shape) for name, value in model.state_dict().items()},
    }


def save_checkpoint(model, tokenizer, global_step, training_state=None, *, settings=None):
    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step must be a non-negative integer")
    compatibility = _compatibility(model, tokenizer, settings)
    run_config = {name: value for name, value in vars(config).items() if name.isupper()}
    run_config.update(settings or {})
    run_config = {name: run_config[name] for name in vars(config) if name.isupper()}
    run_config.update(compatibility["config"])
    fingerprint = _digest(compatibility)
    directory = ROOT / run_config["CHECKPOINT_DIR"] / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{global_step:09d}.pt"
    payload = {
        "compatibility": compatibility,
        "fingerprint": fingerprint,
        "global_step": global_step,
        "model": model.state_dict(),
        "training_state": training_state,
        "run_config": run_config,
    }
    # Publish only complete files, without replacing an existing training step.
    with NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()
    return path


def _read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {
        "compatibility", "fingerprint", "global_step", "model", "training_state", "run_config"
    } <= payload.keys():
        raise ValueError("Checkpoint is missing required metadata")
    saved = payload["compatibility"]
    if _digest(saved) != payload["fingerprint"]:
        raise ValueError("Checkpoint compatibility fingerprint is invalid")
    manager = _manager()
    if saved.get("manager") != manager:
        raise ValueError("Checkpoint manager schema or policy does not match")
    if set(payload["run_config"]) != {name for name in vars(config) if name.isupper()}:
        raise ValueError("Checkpoint run_config is incomplete")
    return payload


def _load_weights(payload, model, tokenizer, settings=None):
    saved = payload["compatibility"]
    expected = _compatibility(model, tokenizer, settings)
    if _digest(saved) != _digest(expected):
        changed = [name for name in expected if _digest(saved.get(name)) != _digest(expected[name])]
        raise ValueError(f"Incompatible checkpoint: {', '.join(changed)}")
    current = model.state_dict()
    weights = payload["model"]
    if weights.keys() != current.keys() or any(
        not isinstance(weights[name], torch.Tensor) or weights[name].shape != value.shape
        for name, value in current.items()
    ):
        raise ValueError("Checkpoint parameter names or shapes do not match the model")
    model.load_state_dict(weights, strict=True)


def load_checkpoint(path, model, tokenizer, *, settings=None):
    """Validate an existing model/tokenizer and return state for training resume."""
    payload = _read_checkpoint(path)
    current = vars(config) | (settings or {})
    changed = [name for name in payload["compatibility"]["manager"]["resume_fields"]
               if _digest(current[name]) != _digest(payload["run_config"][name])]
    if changed:
        raise ValueError(f"Incompatible resume configuration: {', '.join(changed)}; use load_pretrained for fine-tuning")
    _load_weights(payload, model, tokenizer, settings)
    return {"global_step": payload["global_step"], "training_state": payload["training_state"],
            "run_config": payload["run_config"]}


def load_pretrained(path):
    """Rebuild Stage1 offline for fine-tuning; return model, tokenizer, saved config."""
    from tokenizers import AddedToken, Tokenizer
    from transformers import BertConfig, DINOv3ViTConfig, PreTrainedTokenizerFast
    from src.trainingph1.model import ModelStage1

    payload = _read_checkpoint(path)
    saved = payload["compatibility"]
    for name, digest in saved["sources"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Incompatible checkpoint source: {name}")
    for name, saved_version in saved["versions"].items():
        if version(name) != saved_version:
            raise ValueError(f"Incompatible checkpoint dependency: {name}")
    settings = saved["config"]
    model = ModelStage1(
        bert_name=settings["BERT_MODEL_ID"],
        return_layer=settings["VISION_RETURN_LAYER"],
        model_vision_id=settings["VISION_MODEL_ID"],
        num_queries=settings["NUM_QUERIES"],
        cross_attn_every=settings["CROSS_ATTN_EVERY"],
        hidden_dim=settings["HIDDEN_DIM"],
        itc_dim=settings["ITC_DIM"],
        normalize_eps=settings["NORMALIZE_EPS"],
        bert_config=BertConfig.from_dict(saved["qformer_config"]),
        vision_config=DINOv3ViTConfig.from_dict(saved["vision_config"]),
    )
    token_config = saved["tokenizer"]
    added = {token["content"]: AddedToken(**{k: v for k, v in token.items() if k != "id"})
             for token in token_config["backend"]["added_tokens"]}
    specials = {name: [added.get(t, t) for t in value] if isinstance(value, list)
                else added.get(value, value) for name, value in token_config["special_tokens"].items()}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(token_config["backend"])),
        **specials,
        **{k: v for k, v in token_config.items() if k not in {"backend", "special_tokens"}},
    )
    _load_weights(payload, model, tokenizer, settings)
    model.checkpoint_config = settings.copy()
    return model.eval(), tokenizer, settings.copy()
