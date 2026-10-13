"""Detached, count-weighted diagnostics; no additional model forwards or RNG use."""

import math

import torch

RETRIEVAL = ('recall_at_1', 'recall_at_5', 'positive_mass', 'positive_cosine',
             'hard_negative_cosine', 'margin', 'normalized_entropy', 'positive_candidates', 'candidates')
NAMES = tuple(f'itc/{direction}/{name}' for direction in ('i2t', 't2i') for name in RETRIEVAL) + (
    'teacher/image_cosine', 'teacher/text_cosine', 'representation/query_cosine',
    'representation/teacher_text_std', 'data/valid_fraction', 'data/token_utilization',
    'data/itm_eligible_fraction', 'data/tokens_per_sample', 'itc/pseudo_weight',
    'itm/positive/recall', 'itm/positive/probability', 'itm/negative_image/recall',
    'itm/negative_image/probability', 'itm/negative_text/recall', 'itm/negative_text/probability',
    'itg/token_accuracy', 'itg/sequence_accuracy', 'loss/itc', 'loss/itm', 'loss/itg',
)
HISTOGRAMS = ('itc/positive_cosine', 'itm/positive_probability', 'itm/negative_probability')
GRADIENT_GROUPS = ('qformer', 'query_proj', 'text_proj', 'itm_head', 'dec_embedding')


class TrainingDiagnostics:
    def __init__(self, device, max_samples=0):
        self.sums = torch.zeros(len(NAMES), 2, dtype=torch.float64, device=device)
        self.histograms = torch.zeros(len(HISTOGRAMS), 22, dtype=torch.float64, device=device)
        self.samples, self.max_samples = [], max_samples

    def add(self, name, total, count):
        i = NAMES.index(name)
        self.sums[i, 0] += total
        self.sums[i, 1] += count

    def mean(self, name, values):
        self.add(name, values.double().sum(), values.numel())

    def histogram(self, name, values):
        values = values.detach().float().flatten()
        values = values[values.isfinite()]
        low = -1. if name == 'itc/positive_cosine' else 0.
        values = values.clamp(low, 1.)
        row = self.histograms[HISTOGRAMS.index(name)]
        row[:20] += torch.histc(values, bins=20, min=low, max=1.)
        row[20] += values.double().sum()
        row[21] += values.double().square().sum()

    @torch.no_grad()
    def itc(self, itc, online, momentum, keys, queue_ids, ids, valid, temperature, alpha):
        with torch.autocast(ids.device.type, enabled=False):
            bank_ids = torch.cat((keys['image_ids'][keys['valid_mask']], queue_ids))
            positive = ids[valid, None].eq(bank_ids[None, :])
            for direction in ('i2t', 't2i'):
                logits = itc[f'logits_{direction}'].detach().float()
                if not logits.numel():
                    continue
                prefix = f'itc/{direction}'
                top = positive.gather(1, logits.topk(min(5, logits.size(1)), dim=1).indices)
                self.mean(f'{prefix}/recall_at_1', top[:, 0])
                self.mean(f'{prefix}/recall_at_5', top.any(-1))
                log_probability = logits.log_softmax(-1)
                probability = log_probability.exp()
                self.mean(f'{prefix}/positive_mass', (probability * positive).sum(-1))
                positive_cosine = (logits * positive).sum(-1) * temperature / positive.sum(-1).clamp_min(1)
                self.mean(f'{prefix}/positive_cosine', positive_cosine)
                if direction == 'i2t':
                    self.histogram('itc/positive_cosine', positive_cosine)
                has_negative = (~positive).any(-1)
                hardest = logits.masked_fill(positive, -torch.inf).amax(-1)[has_negative] * temperature
                best_positive = logits.masked_fill(~positive, -torch.inf).amax(-1)[has_negative] * temperature
                self.mean(f'{prefix}/hard_negative_cosine', hardest)
                self.mean(f'{prefix}/margin', best_positive - hardest)
                if logits.size(1) > 1:
                    self.mean(f'{prefix}/normalized_entropy',
                              -(probability * log_probability).sum(-1) / math.log(logits.size(1)))
                self.mean(f'{prefix}/positive_candidates', positive.sum(-1))
                self.add(f'{prefix}/candidates', logits.numel(), logits.size(0))
            for kind in ('image', 'text'):
                a, b = online[f'{kind}_features'][valid].float(), momentum[f'{kind}_features'][valid].float()
                self.mean(f'teacher/{kind}_cosine', (a * b).sum(-1))
            queries = online['image_features'][valid].float()
            q = queries.size(1)
            if q > 1:
                self.mean('representation/query_cosine',
                          (queries.sum(1).square().sum(-1) - queries.square().sum((1, 2))) / (q * (q - 1)))
            teacher_text = keys['text_features'][keys['valid_mask']].float()
            if teacher_text.size(0) > 1:
                # Query counts weight the reference-bank statistic across micro-batches.
                self.add('representation/teacher_text_std', teacher_text.std(0, unbiased=False).mean() * valid.sum(),
                         valid.sum())
            self.add('itc/pseudo_weight', valid.sum() * alpha, valid.sum())
            self.add('data/valid_fraction', valid.sum(), valid.numel())

    @torch.no_grad()
    def itm(self, branch, logits, valid):
        logits = logits.detach()[valid].float()
        self.mean(f'itm/{branch}/recall', logits.ge(0) if branch == 'positive' else logits.lt(0))
        self.mean(f'itm/{branch}/probability', logits.sigmoid())
        self.histogram('itm/positive_probability' if branch == 'positive' else 'itm/negative_probability',
                       logits.sigmoid())

    @torch.no_grad()
    def itg(self, logits, labels, valid, images):
        prediction = logits.detach().argmax(-1)
        tokens = valid[:, None] & labels.ne(-100)
        self.mean('itg/token_accuracy', prediction.eq(labels)[tokens])
        self.mean('itg/sequence_accuracy', (prediction.eq(labels) | ~tokens).all(-1)[tokens.any(-1)])
        self.add('data/token_utilization', tokens.sum(), valid.sum() * labels.size(1))
        self.add('data/tokens_per_sample', tokens.sum(), valid.sum())
        if len(self.samples) < self.max_samples:
            for i in tokens.any(-1).nonzero().flatten()[:self.max_samples - len(self.samples)].tolist():
                self.samples.append((images[i].detach().cpu(), labels[i, tokens[i]].cpu(), prediction[i, tokens[i]].cpu()))

    @torch.no_grad()
    def losses(self, losses):
        for name in ('itc', 'itm', 'itg'):
            self.add(f'loss/{name}', losses[f'loss_{name}_sum'].detach(), losses[f'count_{name}'])
        self.add('data/itm_eligible_fraction', losses['count_itm'] / 3, losses['count_itc'])

    def pack(self):
        return torch.cat((self.sums.flatten(), self.histograms.flatten()))

    def summarize(self, settings, packed=None):
        packed = self.pack() if packed is None else packed
        sums = packed[:self.sums.numel()].reshape_as(self.sums).cpu().tolist()
        scalars = {name: total / count for name, (total, count) in zip(NAMES, sums) if count > 0}
        scalars.update({f'counts/{kind}': sums[NAMES.index(name)][1] for kind, name in
                        (('itc_queries', 'loss/itc'), ('itm_pairs', 'loss/itm'), ('itg_tokens', 'loss/itg'))})
        tp, positives = sums[NAMES.index('itm/positive/recall')]
        negatives = sum(sums[NAMES.index(f'itm/{b}/recall')][1] for b in ('negative_image', 'negative_text'))
        tn = sum(sums[NAMES.index(f'itm/{b}/recall')][0] for b in ('negative_image', 'negative_text'))
        if positives and negatives:
            fp, fn = negatives - tn, positives - tp
            scalars.update({'itm/accuracy': (tp + tn) / (positives + negatives),
                            'itm/balanced_accuracy': (tp / positives + tn / negatives) / 2,
                            'itm/precision': tp / max(tp + fp, 1),
                            'itm/f1': 2 * tp / max(2 * tp + fp + fn, 1),
                            'itm/true_positive': tp, 'itm/true_negative': tn,
                            'itm/false_positive': fp, 'itm/false_negative': fn})
        if 'loss/itg' in scalars and scalars['loss/itg'] < 700:
            scalars['itg/perplexity'] = math.exp(scalars['loss/itg'])
        weighted = {name: scalars.get(f'loss/{name}', 0.) * settings[f'{name.upper()}_WEIGHT']
                    for name in ('itc', 'itm', 'itg')}
        for name, value in weighted.items():
            scalars[f'loss/{name}_weighted'] = value
            if sum(weighted.values()) > 0:
                scalars[f'loss/{name}_share'] = value / sum(weighted.values())
        histograms = packed[self.sums.numel():].reshape_as(self.histograms).cpu().tolist()
        return dict(scalars=scalars, histograms=dict(zip(HISTOGRAMS, histograms)), samples=self.samples)
