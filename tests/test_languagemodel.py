import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    BertConfig, DINOv3ViTConfig, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM,
    Trainer, TrainingArguments,
)

from src.languagemodel import QwenLanguageModel
from src.trainingph2.data import ChatCollator, InstructCollator
from src.trainingph2.model import ModelStage2
from src.trainingph1.model import ModelStage1


def tiny_tokenizer():
    backend = Tokenizer(models.WordLevel(
        {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'user': 3, 'assistant': 4,
         'hello': 5, 'world': 6, 'answer': 7, '<think>': 8, '</think>': 9}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       eos_token='[EOS]', unk_token='[UNK]', padding_side='left')
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role + ' ' + m.content + ' [EOS] ' }}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% if not enable_thinking %}<think> </think> {% endif %}{% endif %}")
    return tokenizer


def tiny_stage1():
    return ModelStage1(
        'tiny', -1, 'tiny', 2, 1, 8, 4, 1e-12,
        bert_config=BertConfig(vocab_size=10, hidden_size=8, num_hidden_layers=2,
                               num_attention_heads=2, intermediate_size=16),
        vision_config=DINOv3ViTConfig(hidden_size=8, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=16,
                                     image_size=16, patch_size=8), vision_mock=False)


class LanguageModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.folder = tempfile.TemporaryDirectory()
        cls.base = Path(cls.folder.name) / 'base'
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2))
        model.save_pretrained(cls.base)
        tiny_tokenizer().save_pretrained(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def load(self, **kwargs):
        return QwenLanguageModel.from_pretrained(
            str(self.base), load_in_4bit=False, device='cpu', local_files_only=True,
            attn_implementation='eager', rank=2, alpha=4, lora_dropout=0., **kwargs)

    def batch(self, tokenizer):
        return ChatCollator(tokenizer)([
            {'messages': [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'answer'}]},
            {'messages': [{'role': 'user', 'content': 'hello world'}, {'role': 'assistant', 'content': 'hello world'}]},
        ])

    def test_chat_labels_mask_prompt_padding_but_keep_eos(self):
        tokenizer = tiny_tokenizer()
        for side in ('left', 'right'):
            for thinking in (False, True):
                with self.subTest(side=side, thinking=thinking):
                    tokenizer.padding_side = side
                    batch = ChatCollator(tokenizer, enable_thinking=thinking)([
                        {'messages': [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'answer'}]},
                        {'messages': [{'role': 'user', 'content': 'hello world'}, {'role': 'assistant', 'content': 'hello world'}]},
                    ])
                    self.assertEqual(batch['labels'][0][batch['labels'][0] != -100].tolist(), [7, 2])
                    self.assertEqual(batch['labels'][1][batch['labels'][1] != -100].tolist(), [5, 6, 2])
                    self.assertTrue(batch['labels'][~batch['attention_mask'].bool()].eq(-100).all())
        tokenizer.pad_token = tokenizer.eos_token
        self.assertEqual(self.batch(tokenizer)['labels'].eq(2).sum().item(), 2)

    def test_chat_truncation_preserves_eos_and_rejects_missing_supervision(self):
        tokenizer = tiny_tokenizer()
        record = {'messages': [{'role': 'user', 'content': 'hello'},
                               {'role': 'assistant', 'content': 'answer ' * 20}]}
        batch = ChatCollator(tokenizer, max_length=10)([record])
        self.assertEqual(batch['input_ids'].shape[1], 10)
        self.assertEqual(batch['labels'][0, -1].item(), tokenizer.eos_token_id)
        with self.assertRaisesRegex(ValueError, 'Prompt'):
            ChatCollator(tokenizer, max_length=3)([record])
        for messages in ([{'role': 'user', 'content': 'hello'}],
                         [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': ''}]):
            with self.assertRaises(ValueError):
                ChatCollator(tokenizer)([{'messages': messages}])

    def test_visual_backward_updates_adapter_and_projector_only(self):
        model = self.load(qformer_dim=8, projector_hidden_dim=12)
        self.assertTrue(model.model.is_gradient_checkpointing)
        batch = self.batch(model.tokenizer)
        features = torch.randn(2, 3, 8, requires_grad=True)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
        loss = model(**batch, query_features=features).loss
        self.assertTrue(loss.isfinite())
        loss.backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertGreater(model.projector.fc1.weight.grad.abs().sum().item(), 0)
        self.assertTrue(any('lora_' in name and p.grad is not None and p.grad.abs().sum() > 0
                            for name, p in model.named_parameters()))
        self.assertTrue(all('lora_' in name or name.startswith('projector.')
                            for name, p in model.named_parameters() if p.requires_grad))
        optimizer.step()
        for name, before in frozen.items():
            torch.testing.assert_close(dict(model.named_parameters())[name], before, rtol=0, atol=0)

    def test_prefix_labels_and_loss_match_explicit_causal_reference(self):
        model = self.load(qformer_dim=8, gradient_checkpointing=False).eval()
        batch = self.batch(model.tokenizer)
        features = torch.randn(2, 3, 8)
        inputs = model._inputs(**batch, query_features=features)
        self.assertTrue(inputs['labels'][:, :3].eq(-100).all())
        self.assertTrue(inputs['attention_mask'][:, :3].all())
        output = model(**batch, query_features=features)
        expected = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, 10), inputs['labels'][:, 1:].reshape(-1))
        torch.testing.assert_close(output.loss, expected)
        for row in range(2):
            valid = batch['attention_mask'][row].bool()
            single = model(input_ids=batch['input_ids'][row:row + 1, valid],
                           query_features=features[row:row + 1])
            torch.testing.assert_close(output.logits[row, -1], single.logits[0, -1], rtol=1e-4, atol=1e-5)

    def test_adapter_projector_round_trip_and_frozen_inference(self):
        model = self.load(qformer_dim=8, projector_hidden_dim=12, gradient_checkpointing=False)
        batch = self.batch(model.tokenizer)
        features = torch.randn(2, 3, 8)
        model(**batch, query_features=features).loss.backward()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1)
        optimizer.step()
        model.eval()
        expected = model(**batch, query_features=features).logits.detach()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            self.assertFalse((Path(folder) / 'model.safetensors').exists())
            restored = self.load(adapter_path=folder, trainable=False)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            torch.testing.assert_close(restored(**batch, query_features=features).logits, expected)
            trainable = self.load(adapter_path=folder)
            self.assertTrue(trainable.projector.fc1.weight.requires_grad)
            self.assertTrue(any(p.requires_grad for name, p in trainable.named_parameters() if 'lora_' in name))

    def test_generation_returns_new_tokens_and_restores_mode(self):
        for visual in (False, True):
            with self.subTest(visual=visual):
                model = self.load(qformer_dim=8 if visual else None)
                ids = torch.tensor([[0, 5], [5, 6]])
                features = torch.randn(2, 3, 8) if visual else None
                output = model.generate(ids, ids.ne(0), features, max_new_tokens=2, min_new_tokens=2, do_sample=False)
                self.assertEqual(output.shape, (2, 2))
                self.assertTrue(model.training)
                with patch.object(model.model, 'generate', side_effect=RuntimeError('generation failed')):
                    with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                        model.generate(ids, ids.ne(0), features)
                self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, 'left-padded'):
            model.generate(torch.tensor([[5, 0]]), torch.tensor([[1, 0]]))

    def test_invalid_visual_inputs_and_empty_labels_are_rejected(self):
        model = self.load(qformer_dim=8)
        batch = self.batch(model.tokenizer)
        with self.assertRaisesRegex(ValueError, 'supervised'):
            model(**(batch | {'labels': torch.full_like(batch['labels'], -100)}))
        for features in (torch.randn(1, 3, 8), torch.randn(2, 3, 7), torch.randn(2, 0, 8)):
            with self.assertRaisesRegex(ValueError, 'query_features'):
                model(**batch, query_features=features)
        with self.assertRaisesRegex(ValueError, 'context length'):
            model(input_ids=torch.ones(2, 64, dtype=torch.long), query_features=torch.randn(2, 1, 8))

    def test_qlora_rejects_cpu_before_loading(self):
        with patch('src.languagemodel.AutoModelForCausalLM.from_pretrained') as load:
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                QwenLanguageModel.from_pretrained(device='cpu')
            load.assert_not_called()

    def test_qlora_load_configuration_uses_nf4_and_one_device(self):
        base = Qwen3ForCausalLM.from_pretrained(self.base)
        with patch('torch.cuda.is_available', return_value=True), \
             patch('src.languagemodel.AutoModelForCausalLM.from_pretrained', return_value=base) as load, \
             patch('src.languagemodel.AutoTokenizer.from_pretrained', return_value=tiny_tokenizer()):
            QwenLanguageModel.from_pretrained(device='cuda:1', dtype=torch.bfloat16)
        options = load.call_args.kwargs
        self.assertEqual(options['device_map'], {'': 'cuda:1'})
        quantization = options['quantization_config']
        self.assertTrue(quantization.load_in_4bit)
        self.assertTrue(quantization.bnb_4bit_use_double_quant)
        self.assertEqual(quantization.bnb_4bit_quant_type, 'nf4')
        self.assertEqual(quantization.bnb_4bit_compute_dtype, torch.bfloat16)

    def test_stage1_queries_connect_to_qwen_without_text_embeddings(self):
        stage1 = tiny_stage1()
        llm = self.load(qformer_dim=8)
        features = stage1.encode_image(torch.randn(2, 3, 16, 16))
        queries = stage1.qformer_model(features, None, None, 'itc')['query_output']
        llm(**self.batch(llm.tokenizer), query_features=queries).loss.backward()
        self.assertGreater(stage1.qformer_model.query_tokens.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in stage1.vision_encoder.parameters()))

    def test_instruct_forward_conditions_queries_on_prompt_only(self):
        model = ModelStage2(tiny_stage1(), self.load(qformer_dim=8), tiny_tokenizer()).eval()
        collator = InstructCollator(model.tokenizer, model.qformer_tokenizer)
        records = [{'image': torch.randn(3, 16, 16), 'prompt': 'hello', 'answer': 'answer'}]
        first = collator(records)
        second = collator([records[0] | {'answer': 'world'}])
        torch.testing.assert_close(first['qformer_input_ids'], second['qformer_input_ids'])
        queries = model.encode_image(first['images'], first['qformer_input_ids'], first['qformer_attention_mask'])
        changed = collator([records[0] | {'prompt': 'world'}])
        other = model.encode_image(changed['images'], changed['qformer_input_ids'], changed['qformer_attention_mask'])
        self.assertFalse(torch.allclose(queries, other))
        self.assertTrue(model(**first).loss.isfinite())
        self.assertEqual(first['labels'][first['labels'] != -100].tolist(), [7, 2])

    def test_instruct_training_freezes_vision_and_optionally_qformer(self):
        for train_qformer in (True, False):
            with self.subTest(train_qformer=train_qformer):
                model = ModelStage2(tiny_stage1(), self.load(qformer_dim=8), tiny_tokenizer(), train_qformer)
                batch = InstructCollator(model.tokenizer, model.qformer_tokenizer)([
                    {'image': torch.randn(3, 16, 16), 'prompt': 'hello', 'answer': 'answer'}])
                model(**batch).loss.backward()
                self.assertFalse(model.vision_encoder.training)
                self.assertFalse(model.embeddings.training)
                self.assertEqual(model.qformer.training, train_qformer)
                self.assertTrue(all(p.grad is None for p in model.vision_encoder.parameters()))
                self.assertTrue(all(p.grad is None for p in model.embeddings.parameters()))
                if train_qformer:
                    self.assertGreater(model.qformer.query_tokens.grad.abs().sum().item(), 0)
                    self.assertGreater(model.qformer.layers[0].text_output.dense.weight.grad.abs().sum().item(), 0)
                else:
                    self.assertTrue(all(p.grad is None for p in model.qformer.parameters()))
                self.assertGreater(model.projector.fc1.weight.grad.abs().sum().item(), 0)

    def test_generation_accepts_one_image_and_prompt_and_restores_training(self):
        model = ModelStage2(tiny_stage1(), self.load(qformer_dim=8), tiny_tokenizer())
        image = Image.new('RGB', (24, 20), 'red')
        options = dict(max_new_tokens=2, min_new_tokens=2, do_sample=False)
        answer = model.generation(image, 'hello', **options)
        self.assertIsInstance(answer, str)
        self.assertTrue(model.training)
        self.assertTrue(model.qformer.training)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.png'
            image.save(path)
            self.assertEqual(model.generation(path, 'hello', **options), answer)
        with patch.object(model.language_model, 'generate', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                model.generation(image, 'hello')
        self.assertTrue(model.training)
        self.assertFalse(model.vision_encoder.training)
        with self.assertRaises(ValueError):
            model.generation(image, '')

    def test_stage2_export_restores_qformer_and_projector(self):
        from test_checkpoint_fsdp import fixture, prepare
        from src.trainingph1.engine_checkpoint import save_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            tokenizer = tiny_tokenizer()
            with patch('src.trainingph1.engine_fsdp.ModelStage1', side_effect=lambda *a, **k: tiny_stage1()), \
                 patch('src.trainingph1.engine_fsdp.AutoTokenizer.from_pretrained', return_value=tokenizer):
                state = prepare(fixture(folder))
            source = save_checkpoint(state)
            model = ModelStage2(state.model, self.load(qformer_dim=8), tokenizer, image_size=16)
            batch = InstructCollator(model.tokenizer, tokenizer)([
                {'image': torch.randn(3, 16, 16), 'prompt': 'hello', 'answer': 'answer'}])
            model(**batch).loss.backward()
            torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01).step()
            model.eval()
            expected = model(**batch).logits.detach()
            adapter = Path(folder) / 'adapter'
            model.save_pretrained(adapter)
            restored = ModelStage2.from_stage1(source, adapter_path=adapter, load_in_4bit=False,
                                               device='cpu', trainable=False, attn_implementation='eager')
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            torch.testing.assert_close(restored(**batch).logits, expected)
            self.assertEqual(restored.preprocess_config['image_size'], 16)

    def test_trainer_sft_saves_adapter_and_resumes_optimizer(self):
        records = [{'messages': [{'role': 'user', 'content': 'hello'},
                                  {'role': 'assistant', 'content': 'answer'}]}] * 4
        with tempfile.TemporaryDirectory() as folder:
            def trainer(llm, steps):
                return Trainer(
                    model=llm.model, processing_class=llm.tokenizer,
                    args=TrainingArguments(
                        output_dir=folder, use_cpu=True, max_steps=steps, per_device_train_batch_size=1,
                        gradient_accumulation_steps=2, save_steps=1, report_to='none',
                        remove_unused_columns=False, disable_tqdm=True),
                    train_dataset=records, data_collator=ChatCollator(llm.tokenizer))

            original = trainer(self.load(), 2)
            original.train()
            checkpoint = Path(folder) / 'checkpoint-1'
            self.assertTrue((checkpoint / 'adapter_model.safetensors').exists())
            self.assertTrue((checkpoint / 'optimizer.pt').exists())
            self.assertFalse((checkpoint / 'model.safetensors').exists())
            resumed = trainer(self.load(adapter_path=str(checkpoint)), 2)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 2)
            self.assertEqual(resumed.lr_scheduler.state_dict(), original.lr_scheduler.state_dict())
            for name, parameter in original.model.named_parameters():
                torch.testing.assert_close(parameter, dict(resumed.model.named_parameters())[name], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA for real NF4 training')
    def test_cuda_nf4_backward(self):
        model = QwenLanguageModel.from_pretrained(str(self.base), device='cuda:0', qformer_dim=8, rank=2)
        self.assertTrue(model.model.is_loaded_in_4bit)
        loss = model(**self.batch(model.tokenizer), query_features=torch.randn(2, 3, 8)).loss
        self.assertTrue(loss.isfinite())
        loss.backward()
        self.assertGreater(model.projector.fc1.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
