import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage1Criterion(nn.Module):
    def get_itm_loss(self, itm_logits, labels):
        return {"loss_itm": F.binary_cross_entropy_with_logits(itm_logits.float(), labels.float())}

    def get_itg_loss(self, itg_logits, labels):
        loss = F.cross_entropy(itg_logits.float().reshape(-1, itg_logits.size(-1)),
                               labels.reshape(-1), ignore_index=-100, reduction="sum")
        return {"loss_itg": loss / labels.ne(-100).sum().clamp_min(1)}

    def get_itc_loss(self, image_features, text_features, momentum_image_features,
                     momentum_text_features, img_queue, text_queue, temperature, pseudo_weight,
                     image_ids, queue_ids):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= pseudo_weight <= 1:
            raise ValueError("pseudo_weight must be in [0, 1]")

        with torch.autocast(device_type=image_features.device.type, enabled=False):
            def similarity(images, texts):
                return torch.einsum("bqd,kd->bkq", images, texts).amax(-1) / temperature

            with torch.no_grad():
                image_keys = momentum_image_features.detach().float()
                text_keys = momentum_text_features.detach().float()
                image_bank = torch.cat((image_keys, img_queue.detach().float()), dim=0)
                text_bank = torch.cat((text_keys, text_queue.detach().float()), dim=0)
                bank_ids = torch.cat((image_ids, queue_ids))
                targets = (image_ids[:, None] == bank_ids[None, :]).float()
                targets = targets / targets.sum(-1, keepdim=True)
                target_i2t = (1 - pseudo_weight) * targets + pseudo_weight * similarity(image_keys, text_bank).softmax(-1)
                target_t2i = (1 - pseudo_weight) * targets + pseudo_weight * similarity(image_bank, text_keys).T.softmax(-1)

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