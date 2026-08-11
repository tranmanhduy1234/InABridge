"""Default architecture configuration for InA-Bridge."""

import torch


IMAGE_ENCODER_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMAGE_SIZE = 448
CENTER_CROP = False
RETURN_LAYER = -2
IMAGE_ENCODER_OUT_DIMENSION = 1024

QFORMER_TOKENIZER_ID = "microsoft/deberta-v3-base"
NUM_QUERIES = 32

PROJECTOR_IN_DIMENSION = 768
PROJECTOR_HIDDEN_DIMENSION = 1536
PROJECTOR_OUT_DIMENSION = 2560

LLM_MODEL_ID = "Qwen/Qwen3-4B"

COMPUTE_TYPE = torch.bfloat16
