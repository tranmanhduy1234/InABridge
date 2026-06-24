import torch
import torch.nn as nn
from src.component.ImageEncoder.imageEncoder import ImageEncoder
from src.config import *
class InABridgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.imageEncoder = ImageEncoder(model_id=IMAGE_ENCODER_MODEL_ID, return_layer=RETURN_LAYER)
        self.Bridge = None
        self.multiLayerPerceptron = None
        self.llmModel = None