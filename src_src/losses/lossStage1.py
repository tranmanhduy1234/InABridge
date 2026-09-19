import math

import torch
import torch.nn.functional as F

from src_src import config_model as cfg

def calculate_loss_ph1(model, momentum_model, queue, image_tensor, input_ids, attention_mask,
                       *, distill_weight, temperature=cfg.ITC_TEMPERATURE,
                       key_image_tensor=None, update_queue=cfg.UPDATE_QUEUE,
                       itc_weight=cfg.ITC_LOSS_WEIGHT, itm_weight=cfg.ITM_LOSS_WEIGHT,
                       itg_weight=cfg.ITG_LOSS_WEIGHT, ignore_index=cfg.ITG_IGNORE_INDEX):
    batch = image_tensor.size(0)
    if batch < 2 or input_ids.size(0) != batch or attention_mask.shape != input_ids.shape:
        raise ValueError("expected at least two matching image/text samples and a matching attention mask")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= distill_weight <= 1:
        raise ValueError("distill_weight must be in [0, 1]")
    if any(not math.isfinite(w) or w < 0 for w in (itc_weight, itm_weight, itg_weight)):
        raise ValueError("loss weights must be finite and non-negative")
    if key_image_tensor is None:
        key_image_tensor = image_tensor
    if key_image_tensor.shape != image_tensor.shape:
        raise ValueError("query and key image batches must have the same shape")

    image_features = model.encode_image(image_tensor)
    online = model(image_features, input_ids, attention_mask, objective="itc")
    images, text_features = online["image_features"].float(), online["text_features"].float()
    with torch.no_grad():
        key_features = momentum_model.encode_image(key_image_tensor)
        momentum = momentum_model(key_features, input_ids, attention_mask, objective="itc")
        image_keys = momentum["image_features"].detach().float()
        text_keys = momentum["text_features"].detach().float()
    old_images, old_texts = queue.get()
    image_bank = torch.cat((image_keys, old_images.float()), dim=0)
    text_bank = torch.cat((text_keys, old_texts.float()), dim=0)

    def similarity(image, text):
        return torch.einsum("bqd,kd->bkq", image, text).amax(-1) / temperature

    sim_i2t = similarity(images, text_bank)
    sim_t2i = similarity(image_bank, text_features).T
    with torch.no_grad():
        targets = torch.zeros_like(sim_i2t)
        targets[:, :batch].fill_diagonal_(1)
        target_i2t = (1 - distill_weight) * targets + distill_weight * similarity(image_keys, text_bank).softmax(-1)
        target_t2i = (1 - distill_weight) * targets + distill_weight * similarity(image_bank, text_keys).T.softmax(-1)
        diagonal = torch.eye(batch, dtype=torch.bool, device=image_tensor.device)
        sim_batch = similarity(images.detach(), text_features.detach())
        # Mask positives before softmax so a dominant positive cannot cause underflow.
        weights_i2t = sim_batch.masked_fill(diagonal, -torch.inf).softmax(-1)
        weights_t2i = sim_batch.T.masked_fill(diagonal, -torch.inf).softmax(-1)
        neg_text = torch.multinomial(weights_i2t, 1).squeeze(1)
        neg_image = torch.multinomial(weights_t2i, 1).squeeze(1)

    loss_i2t = -(target_i2t * sim_i2t.log_softmax(-1)).sum(-1).mean()
    loss_t2i = -(target_t2i * sim_t2i.log_softmax(-1)).sum(-1).mean()
    loss_itc = (loss_i2t + loss_t2i) / 2

    itm_images = torch.cat((image_features, image_features, image_features[neg_image]), dim=0)
    itm_ids = torch.cat((input_ids, input_ids[neg_text], input_ids), dim=0)
    itm_mask = torch.cat((attention_mask, attention_mask[neg_text], attention_mask), dim=0)
    itm = model(itm_images, itm_ids, itm_mask, objective="itm")
    itm_logits = itm["itm_logits"].float()
    itm_labels = torch.cat((
        torch.ones(batch, dtype=torch.long, device=itm_logits.device),
        torch.zeros(2 * batch, dtype=torch.long, device=itm_logits.device),
    ))
    loss_itm = F.cross_entropy(itm_logits, itm_labels)

    itg = model(image_features, input_ids, attention_mask, objective="itg")
    itg_labels = input_ids[:, 1:].masked_fill(~attention_mask[:, 1:].bool(), ignore_index)
    loss_itg = F.cross_entropy(
        itg["itg_logits"].float().reshape(-1, itg["itg_logits"].size(-1)),
        itg_labels.reshape(-1), ignore_index=ignore_index,
    )
    if model.training and update_queue:
        queue.enqueue(image_keys, text_keys)
    return {
        "loss": itc_weight * loss_itc + itm_weight * loss_itm + itg_weight * loss_itg,
        "loss_itc": loss_itc, "loss_itm": loss_itm, "loss_itg": loss_itg,
        "loss_i2t": loss_i2t, "loss_t2i": loss_t2i,
        "logits_i2t": sim_i2t, "logits_t2i": sim_t2i,
        "itm_logits": itm_logits, "itm_labels": itm_labels,
        "neg_image_indices": neg_image, "neg_text_indices": neg_text,
        "image_keys": image_keys, "text_keys": text_keys,
    }
