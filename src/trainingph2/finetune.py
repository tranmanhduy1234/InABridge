"""Text QLoRA SFT; visual stage 2 can reuse the LLM's query_features interface."""

import argparse
import json
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed

from src.languagemodel import QwenLanguageModel
from src.trainingph2.data import ChatCollator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, help='JSON array of {messages: [...]} records')
    parser.add_argument('--validation')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--adapter', help='Initialize from adapter weights; starts a new optimizer')
    parser.add_argument('--resume', help='Trainer checkpoint containing optimizer/scheduler/RNG')
    parser.add_argument('--epochs', type=float, default=3.)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=32)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable-thinking', action='store_true')
    args = parser.parse_args()
    if args.adapter and args.resume:
        parser.error('--adapter and --resume are mutually exclusive')
    set_seed(args.seed)
    llm = QwenLanguageModel.from_pretrained(args.model, adapter_path=args.adapter or args.resume,
                                          rank=args.rank, alpha=args.alpha)
    if llm.projector is not None:
        parser.error('This entry point trains text adapters; use query_features for visual stage 2')

    def read(path):
        records = json.loads(Path(path).read_text())
        if not isinstance(records, list) or not records:
            raise ValueError('Dataset must be a nonempty JSON array')
        return records

    training = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, bf16=torch.cuda.is_bf16_supported(),
        optim='paged_adamw_8bit', lr_scheduler_type='cosine', warmup_ratio=.03,
        max_grad_norm=1., logging_steps=10, save_steps=args.save_steps, save_total_limit=2,
        eval_strategy='steps' if args.validation else 'no', eval_steps=args.save_steps,
        prediction_loss_only=True, remove_unused_columns=False, report_to='none',
        ddp_find_unused_parameters=False, seed=args.seed)
    trainer = Trainer(
        model=llm.model, args=training, processing_class=llm.tokenizer,
        train_dataset=read(args.train), eval_dataset=read(args.validation) if args.validation else None,
        data_collator=ChatCollator(llm.tokenizer, args.max_length, args.enable_thinking))
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(Path(args.output) / 'adapter'))
    trainer.save_state()


if __name__ == '__main__':
    main()
