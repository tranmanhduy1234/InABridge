import math
import unittest

import torch
import torch.nn.functional as F

from src.utils.training_diagnostics import TrainingDiagnostics


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_multi_positive_retrieval_confusion_and_token_metrics(self):
        diagnostics = TrainingDiagnostics('cpu', max_samples=1)
        ids, valid = torch.tensor([1, 1, 2]), torch.tensor([True, True, False])
        features = F.normalize(torch.arange(1., 10.).reshape(3, 3), dim=-1)
        online = dict(image_features=features[:, None].expand(-1, 2, -1), text_features=features)
        keys = online | dict(image_ids=ids, valid_mask=torch.ones(3, dtype=torch.bool))
        logits = torch.tensor([[4., 3., 1.], [3., 4., 1.]])
        diagnostics.itc(dict(logits_i2t=logits, logits_t2i=logits), online, online, keys,
                        torch.empty(0, dtype=torch.long), ids, valid, .1, .4)
        diagnostics.itm('positive', torch.tensor([2., -2., 100.]), valid)
        diagnostics.itm('negative_image', torch.tensor([-2., -2., 100.]), valid)
        diagnostics.itm('negative_text', torch.tensor([-2., 2., 100.]), valid)
        labels = torch.tensor([[1, 2, -100], [2, 3, 1], [1, 1, 1]])
        prediction = torch.tensor([[1, 0, 0], [2, 3, 1], [0, 0, 0]])
        logits = F.one_hot(prediction, 4).float() * 2
        diagnostics.itg(logits, labels, valid, torch.zeros(3, 3, 2, 2))
        token_mask = valid[:, None] & labels.ne(-100)
        loss_itg = F.cross_entropy(logits[token_mask], labels[token_mask], reduction='sum')
        diagnostics.losses(dict(loss_itc_sum=torch.tensor(2.), count_itc=torch.tensor(2),
                                loss_itm_sum=torch.tensor(6.), count_itm=torch.tensor(6),
                                loss_itg_sum=loss_itg, count_itg=token_mask.sum()))
        result = diagnostics.summarize(dict(ITC_WEIGHT=1., ITM_WEIGHT=2., ITG_WEIGHT=1.))
        values = result['scalars']
        self.assertEqual(values['itc/i2t/recall_at_1'], 1.)
        self.assertEqual(values['itc/t2i/positive_candidates'], 2.)
        self.assertEqual(values['itc/i2t/candidates'], 3.)
        self.assertAlmostEqual(values['itc/i2t/margin'], .3)
        self.assertAlmostEqual(values['representation/query_cosine'], 1., places=6)
        self.assertEqual(values['itm/balanced_accuracy'], .625)
        self.assertEqual(values['itm/precision'], .5)
        self.assertEqual(values['itm/f1'], .5)
        self.assertEqual(values['itm/true_negative'], 3.)
        self.assertEqual(values['itm/false_positive'], 1.)
        self.assertEqual(values['itg/token_accuracy'], .8)
        self.assertEqual(values['itg/sequence_accuracy'], .5)
        self.assertAlmostEqual(values['itg/perplexity'], math.exp(loss_itg.item() / 5))
        self.assertEqual(values['counts/itg_tokens'], 5)
        self.assertEqual(sum(result['histograms']['itm/positive_probability'][:20]), 2)
        self.assertEqual(sum(result['histograms']['itm/negative_probability'][:20]), 4)
        self.assertEqual(len(result['samples']), 1)
        self.assertEqual(result['samples'][0][1].tolist(), [1, 2])
        self.assertFalse(diagnostics.sums.requires_grad)

    def test_count_weighting_and_empty_metrics(self):
        diagnostics = TrainingDiagnostics('cpu')
        diagnostics.add('itg/token_accuracy', 1., 1)
        diagnostics.add('itg/token_accuracy', 0., 9)
        values = diagnostics.summarize(dict(ITC_WEIGHT=1., ITM_WEIGHT=1., ITG_WEIGHT=1.))['scalars']
        self.assertEqual(values['itg/token_accuracy'], .1)
        self.assertNotIn('itm/balanced_accuracy', values)
        self.assertNotIn('itc/i2t/recall_at_1', values)


if __name__ == '__main__':
    unittest.main()
