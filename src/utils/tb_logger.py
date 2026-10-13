"""TensorBoard metrics with explicitly controlled output."""

import json
import math
from pathlib import Path


class TensorBoardLogger:
    def __init__(self, run_dir, *, enabled=True, resume_step=None, flush_secs=30):
        if type(flush_secs) is not int or flush_secs < 1:
            raise ValueError('TENSORBOARD_FLUSH_SECS must be a positive integer')
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(Path(run_dir) / 'tensorboard'), flush_secs=flush_secs,
                                        purge_step=None if resume_step is None else resume_step + 1)
            panels = {
                'ITC retrieval': ('itc/i2t/recall_at_1', 'itc/t2i/recall_at_1', 'itc/i2t/recall_at_5', 'itc/t2i/recall_at_5'),
                'ITC separation': ('itc/i2t/positive_cosine', 'itc/i2t/hard_negative_cosine', 'itc/i2t/margin'),
                'ITM classification': ('itm/balanced_accuracy', 'itm/positive/recall', 'itm/negative_image/recall',
                                       'itm/negative_text/recall', 'itm/f1'),
                'ITG teacher-forced': ('itg/token_accuracy', 'itg/sequence_accuracy', 'itg/perplexity'),
                'Representation': ('teacher/image_cosine', 'teacher/text_cosine', 'representation/query_cosine',
                                   'representation/teacher_text_std'),
                'Loss balance': ('loss/itc_share', 'loss/itm_share', 'loss/itg_share'),
                'Data coverage': ('data/valid_fraction', 'data/token_utilization', 'data/itm_eligible_fraction'),
                'Gradient flow': tuple(f'gradients/{name}/norm' for name in
                                       ('qformer', 'query_proj', 'text_proj', 'itm_head', 'dec_embedding'))
                                 + ('gradients/clip_scale',),
            }
            self.writer.add_custom_scalars({panel: {name: ['Multiline', [f'{split}/diagnostics/{name}'
                                                     for split in (('train',) if name.startswith('gradients/')
                                                                   else ('train', 'validation'))]] for name in names}
                                            for panel, names in panels.items()})

    def log_config(self, settings):
        if self.writer:
            self.writer.add_text('run/config', '```json\n' + json.dumps(settings, indent=2, default=str) + '\n```', 0)

    def log_scalars(self, namespace, values, step):
        if self.writer:
            for name, value in values.items():
                value = float(value)
                if math.isfinite(value):
                    self.writer.add_scalar(f'{namespace}/{name}', value, step)

    def log_diagnostics(self, split, diagnostics, step):
        if not self.writer:
            return
        self.log_scalars(f'{split}/diagnostics', diagnostics['scalars'], step)
        for name, values in diagnostics['histograms'].items():
            counts, total, squared = values[:20], values[20], values[21]
            if not sum(counts):
                continue
            low = -1. if name == 'itc/positive_cosine' else 0.
            self.writer.add_histogram_raw(f'{split}/distributions/{name}', min=low, max=1.,
                                          num=sum(counts), sum=total, sum_squares=squared,
                                          bucket_limits=[low + (1. - low) * i / 20 for i in range(1, 21)],
                                          bucket_counts=counts, global_step=step)

    def log_examples(self, samples, tokenizer, normalization, step):
        if not self.writer or not samples:
            return
        def decode(ids):
            return tokenizer.decode(ids.tolist(), skip_special_tokens=False).replace('|', '\\|').replace('\n', ' ')
        rows = ['Teacher-forced predictions: each token sees the correct preceding tokens; these are not generated captions.',
                '', '| Image | Target tokens | Predicted tokens |', '| --- | --- | --- |']
        for i, (image, labels, prediction) in enumerate(samples):
            if normalization is not None:
                mean, std = (image.new_tensor(x)[:, None, None] for x in normalization)
                image = image * std + mean
            self.writer.add_image(f'validation/examples/image_{i}', image.clamp(0, 1), step)
            rows.append(f'| {i} | {decode(labels)} | {decode(prediction)} |')
        self.writer.add_text('validation/examples/teacher_forced', '\n'.join(rows), step)

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None
