import math

import torch
import torch.nn as nn

class Stage1Criterion(nn.Module):
    def __init__(self, itc_weight, itm_weight, itg_weight, pseudo_weight):
        super().__init__()
        self.itc_weight = itc_weight
        self.itm_weight = itm_weight
        self.itg_weight = itg_weight
        self.pseudo_weight = pseudo_weight

    def get_itc_loss(self, image_features, text_features, momentum_image_features,
                     momentum_text_features, img_queue, text_queue, temperature):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= self.pseudo_weight <= 1:
            raise ValueError("pseudo_weight must be in [0, 1]")

        def similarity(images, texts):
            return torch.einsum("bqd,kd->bkq", images, texts).amax(-1) / temperature

        with torch.no_grad():
            image_keys = momentum_image_features.detach().float()
            text_keys = momentum_text_features.detach().float()
            image_bank = torch.cat((image_keys, img_queue.detach().float()), dim=0)
            text_bank = torch.cat((text_keys, text_queue.detach().float()), dim=0)
            targets = torch.zeros(image_keys.size(0), text_bank.size(0), device=image_keys.device)
            targets.fill_diagonal_(1)
            target_i2t = (1 - self.pseudo_weight) * targets + self.pseudo_weight * similarity(image_keys, text_bank).softmax(-1)
            target_t2i = (1 - self.pseudo_weight) * targets + self.pseudo_weight * similarity(image_bank, text_keys).T.softmax(-1)

        logits_i2t = similarity(image_features.float(), text_bank)
        logits_t2i = similarity(image_bank, text_features.float()).T
        loss_i2t = -(target_i2t * logits_i2t.log_softmax(-1)).sum(-1).mean()
        loss_t2i = -(target_t2i * logits_t2i.log_softmax(-1)).sum(-1).mean()
        return {
            "loss_itc": (loss_i2t + loss_t2i) / 2,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "logits_i2t": logits_i2t,
            "logits_t2i": logits_t2i,
        }
