"""Assistant-only supervision for a final answer, preserving Qwen's chat template."""

import torch


class ChatCollator:
    def __init__(self, tokenizer, max_length=2048, enable_thinking=False):
        if max_length < 2:
            raise ValueError('max_length must be at least 2')
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError('Chat training requires PAD and EOS tokens')
        self.tokenizer, self.max_length, self.enable_thinking = tokenizer, max_length, enable_thinking

    def __call__(self, records):
        examples, labels = [], []
        for record in records:
            messages = record['messages']
            if len(messages) < 2 or messages[-1]['role'] != 'assistant':
                raise ValueError('messages must end with an assistant answer after a prompt')
            prompt = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
            # Rendering the prompt separately also preserves Qwen's empty thinking block.
            answer = messages[-1]['content']
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError('Assistant answer must be nonempty text')
            prefix = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            target = self.tokenizer(answer, add_special_tokens=False)['input_ids'] + [self.tokenizer.eos_token_id]
            if len(prefix) >= self.max_length - 1:
                raise ValueError('Prompt leaves no space for answer and EOS; increase max_length')
            target = target[:self.max_length - len(prefix)]
            target[-1] = self.tokenizer.eos_token_id
            examples.append(dict(input_ids=prefix + target, attention_mask=[1] * (len(prefix) + len(target))))
            labels.append([-100] * len(prefix) + target)
        batch = self.tokenizer.pad(examples, padding=True, return_tensors='pt')
        length = batch['input_ids'].shape[1]
        batch['labels'] = torch.tensor([
            [-100] * (length - len(row)) + row if self.tokenizer.padding_side == 'left'
            else row + [-100] * (length - len(row)) for row in labels], dtype=torch.long)
        return batch


class InstructCollator:
    """Records: transformed image tensor, prompt, answer; only prompt reaches Q-Former."""

    def __init__(self, tokenizer, qformer_tokenizer, max_length=2048, qformer_max_length=128,
                 enable_thinking=False):
        self.chat = ChatCollator(tokenizer, max_length, enable_thinking)
        self.qformer_tokenizer, self.qformer_max_length = qformer_tokenizer, qformer_max_length

    def __call__(self, records):
        prompts = [record['prompt'] for record in records]
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError('Each prompt must be nonempty text')
        batch = self.chat([{'messages': [{'role': 'user', 'content': record['prompt']},
                                         {'role': 'assistant', 'content': record['answer']}]}
                           for record in records])
        instructions = self.qformer_tokenizer(prompts, padding=True, truncation=True,
                                              max_length=self.qformer_max_length, return_tensors='pt')
        batch.update(images=torch.stack([record['image'] for record in records]),
                     qformer_input_ids=instructions['input_ids'],
                     qformer_attention_mask=instructions['attention_mask'])
        return batch
