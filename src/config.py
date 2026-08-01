# IMAGE ENCODER CONFIG (DINOv3 for InA-Bridge)
IMAGE_ENCODER_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMAGE_SIZE2MODEL = 448
DO_CENTER_CROP = False
RETURN_LAYER = -2
IMAGE_ENCODER_OUT_DIMENSION = 1024

# QFormer Config
Q_FORMER_LOAD_MODEL="microsoft/deberta-v3-base"
NUM_QUERIES = 32

# Multi Layer Perception Config
MLP_IN_DIMENSION = 768
MLP_OUT_DIMENSION = 2560
MLP_HIDDEN_DIM = 1536

# LLM Config
LLM_MODL_ID = "Qwen/Qwen3-4B"
LLM_IN_DIMENSION = 2560

# TRAINING
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = torch.bfloat16