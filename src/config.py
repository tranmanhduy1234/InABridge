# Component model
# IMAGE ENCODER CONFIG
IMAGE_ENCODER_MODEL_ID = "facebook/dinov2-large"
IMAGE_SIZE2MODEL = 448
DO_CENTER_CROP = False
RETURN_LAYER = -2


Q_FORMER_LOAD_MODEL="microsoft/deberta-v3-base"
LLM_MODL_ID = "Qwen/Qwen2.5-7B-Instruct"

# from transformers import AutoModel
# model = AutoModel.from_pretrained("facebook/dinov2-large")
# print(model)