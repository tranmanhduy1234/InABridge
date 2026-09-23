import unittest

import torch

from src.losses.lossph1 import Stage1Criterion


class LossTests(unittest.TestCase):
    def setUp(self):
        self.criterion = Stage1Criterion()

    def test_itc_distributes_targets_across_same_ids_in_batch_and_queue(self):
        torch.manual_seed(5)
        features = [torch.nn.functional.normalize(torch.randn(*shape), dim=-1) for shape in
                    [(3, 2, 4), (3, 4), (3, 2, 4), (3, 4), (2, 2, 4), (2, 4)]]
        image_ids, queue_ids = torch.tensor([10, 10, 20]), torch.tensor([10, 30])
        hard = torch.tensor([[1/3, 1/3, 0., 1/3, 0.], [1/3, 1/3, 0., 1/3, 0.], [0., 0., 1., 0., 0.]])
        for alpha in (0., .4):
            out = self.criterion.get_itc_loss(*features, .07, alpha, image_ids, queue_ids)
            sim = lambda images, texts: torch.einsum('bqd,kd->bkq', images, texts).amax(-1) / .07
            teacher_i2t = sim(features[2], torch.cat((features[3], features[5]))).softmax(-1)
            teacher_t2i = sim(torch.cat((features[2], features[4])), features[3]).T.softmax(-1)
            targets = [(1-alpha)*hard + alpha*t for t in (teacher_i2t, teacher_t2i)]
            expected = sum(-(t * out[k].log_softmax(-1)).sum(-1).mean()
                           for t, k in zip(targets, ('logits_i2t', 'logits_t2i'))) / 2
            torch.testing.assert_close(out['loss_itc'], expected)

    def test_itc_disables_autocast_for_similarity_and_loss(self):
        torch.manual_seed(3)
        features = [torch.nn.functional.normalize(torch.randn(*shape), dim=-1) for shape in
                    [(2, 2, 4), (2, 4), (2, 2, 4), (2, 4), (0, 2, 4), (0, 4)]]
        args = (*features, .07, .4, torch.arange(2), torch.empty(0, dtype=torch.long))
        expected = self.criterion.get_itc_loss(*args)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = self.criterion.get_itc_loss(*args)
        for name, value in actual.items():
            self.assertEqual(value.dtype, torch.float32)
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_itm_uses_supplied_pair_labels(self):
        logits = torch.tensor([-3., 1., 2.], requires_grad=True)
        labels = torch.tensor([0, 1, 0])
        loss = self.criterion.get_itm_loss(logits, labels)['loss_itm']
        expected = -(labels * logits.sigmoid().log() + (1 - labels) * (-logits).sigmoid().log()).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(logits.grad.isfinite().all())
        self.assertTrue((logits.grad[labels == 1] < 0).all())
        self.assertTrue((logits.grad[labels == 0] > 0).all())

    def test_itg_uses_aligned_labels_and_ignores_padding(self):
        logits = torch.tensor([[[1., 2., 3.], [3., 2., 1.], [0., 1., 0.]],
                               [[2., 0., 1.], [0., 3., 1.], [1., 0., 2.]]], requires_grad=True)
        input_ids = torch.tensor([[0, 1, 2, 0], [0, 2, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        labels = input_ids.masked_fill(~mask.bool(), -100)[:, 1:]
        loss = self.criterion.get_itg_loss(logits, labels)['loss_itg']
        scores = logits.log_softmax(-1)
        expected = -(scores[0, 0, 1] + scores[0, 1, 2] + scores[0, 2, 0] + scores[1, 0, 2]) / 4
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(logits.grad.isfinite().all())
        self.assertEqual(logits.grad[1, 1:].count_nonzero().item(), 0)
        self.assertLess(logits.grad[0, 2, 0].item(), 0)

    def test_itg_without_valid_tokens_returns_differentiable_zero(self):
        for length in (0, 3):
            with self.subTest(length=length):
                logits = torch.zeros(2, length, 4, requires_grad=True)
                labels = torch.full((2, length), -100, dtype=torch.long)
                loss = self.criterion.get_itg_loss(logits, labels)['loss_itg']
                self.assertEqual(loss.item(), 0)
                loss.backward()
                self.assertEqual(logits.grad.count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
