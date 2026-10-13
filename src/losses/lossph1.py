import math

import torch
from torch import nn
import torch.nn.functional as F

class Stage1Criterion(nn.Module):
    @staticmethod
    def _valid_mask(mask, reference, name="valid_mask"):
        if mask is None:
            print(f"WARNING: {name}=None; defaulting to all True. "
                  "When using FSDP, pass a mask to exclude padded samples from the loss.")
            return torch.ones(reference.size(0), dtype=torch.bool, device=reference.device)
        return mask

    def get_itm_loss(self, itm_logits, labels, valid_mask=None):
        valid_mask = self._valid_mask(valid_mask, itm_logits)
        loss = F.binary_cross_entropy_with_logits(
            itm_logits[valid_mask].float(), labels[valid_mask].float(), reduction="sum")
        return {"loss_itm_sum": loss, "count_itm": valid_mask.sum()}

    def get_itg_loss(self, itg_logits, labels, valid_mask=None):
        valid_mask = self._valid_mask(valid_mask, itg_logits)
        valid = valid_mask[:, None] & labels.ne(-100)
        loss = F.cross_entropy(itg_logits[valid].float(), labels[valid], reduction="sum")
        return {"loss_itg_sum": loss, "count_itg": valid.sum()}

    def get_itc_loss(self, image_features, text_features, momentum_image_features,
                     momentum_text_features, img_queue, text_queue, temperature, pseudo_weight,
                     image_ids, queue_ids, valid_mask=None, *, global_image_keys, global_text_keys,
                     global_image_ids, global_valid_mask=None):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= pseudo_weight <= 1:
            raise ValueError("pseudo_weight must be in [0, 1]")

        valid_mask = self._valid_mask(valid_mask, image_features)
        global_valid_mask = self._valid_mask(global_valid_mask, global_image_keys, "global_valid_mask")
        with torch.autocast(device_type=image_features.device.type, enabled=False):
            def similarity(images, texts):
                return torch.einsum("bqd,kd->bkq", images, texts).amax(-1) / temperature

            with torch.no_grad():
                image_queries = momentum_image_features[valid_mask].detach().float()
                text_queries = momentum_text_features[valid_mask].detach().float()
                image_bank = torch.cat((global_image_keys[global_valid_mask].detach().float(),
                                        img_queue.detach().float()))
                text_bank = torch.cat((global_text_keys[global_valid_mask].detach().float(),
                                       text_queue.detach().float()))
                bank_ids = torch.cat((global_image_ids[global_valid_mask], queue_ids))
                targets = (image_ids[valid_mask, None] == bank_ids[None, :]).float()
                positives = targets.sum(-1, keepdim=True)
                if (positives == 0).any():
                    raise ValueError("Every valid query must have a positive in the candidate bank")
                targets = targets / positives
                target_i2t = ((1 - pseudo_weight) * targets
                              + pseudo_weight * similarity(image_queries, text_bank).softmax(-1))
                target_t2i = ((1 - pseudo_weight) * targets
                              + pseudo_weight * similarity(image_bank, text_queries).T.softmax(-1))

            logits_i2t = similarity(image_features[valid_mask].float(), text_bank)
            logits_t2i = similarity(image_bank, text_features[valid_mask].float()).T
            loss_i2t = -(target_i2t * logits_i2t.log_softmax(-1)).sum()
            loss_t2i = -(target_t2i * logits_t2i.log_softmax(-1)).sum()
            return {
                "loss_itc_sum": (loss_i2t + loss_t2i) / 2,
                "count_itc": valid_mask.sum(),
                "loss_i2t_sum": loss_i2t,
                "loss_t2i_sum": loss_t2i,
                "logits_i2t": logits_i2t,
                "logits_t2i": logits_t2i,
            }
