import json
import os
from pathlib import Path

import torch
import torch.nn as nn
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
