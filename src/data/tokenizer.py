"""Khám phá tokenizer phase 1; không thay đổi tokenizer/model của training.

python -m src.data.tokenizer --local-files-only --max-length 16
Mặc định kiểm tra CLS/SEP như dataloader hiện tại.
Thêm --demo-dec --inspect-embeddings để thử DEC và resize BERT trong RAM.
DEC: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py
"""

import argparse
from copy import deepcopy
import json

import torch
from transformers import AutoConfig, AutoTokenizer, BertModel

from src import config
from src.qformer import generate_mask_qformer


def tokenize_vlm_batch(tokenizer, texts, max_length):
    """Dùng cùng thiết lập tokenization với VLMDataCollator, kiểm tra biên trước PAD."""
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError('Tokenizer phải có CLS và SEP')
    if tokenizer.padding_side != 'right':
        raise ValueError('Pipeline lấy CLS tại vị trí 0, yêu cầu right padding')
    batch = tokenizer(texts, padding=config.TOKENIZER_PADDING,
                      truncation=config.TOKENIZER_TRUNCATION,
                      max_length=max_length, return_tensors='pt')
    ids, mask = batch['input_ids'], batch['attention_mask'].bool()
    for i, (row, valid) in enumerate(zip(ids, mask)):
        tokens = row[valid]
        if len(tokens) < 2 or tokens[0].item() != tokenizer.cls_token_id:
            raise ValueError(f'Mẫu {i} không bắt đầu bằng CLS')
        if tokens[-1].item() != tokenizer.sep_token_id:
            raise ValueError(f'Mẫu {i} không kết thúc bằng SEP trước padding')
        if not row[~valid].eq(tokenizer.pad_token_id).all():
            raise ValueError(f'Mẫu {i} có vị trí bị mask nhưng không phải PAD')
    batch['attention_mask'] = mask
    return batch


def show_vlm_batch(tokenizer, texts, max_length):
    print('\n=== Batch theo dataloader VLM hiện tại: CLS → nội dung → SEP → PAD ===')
    print('padding:', config.TOKENIZER_PADDING, '| truncation:', config.TOKENIZER_TRUNCATION)
    batch = tokenize_vlm_batch(tokenizer, texts, max_length)
    for i, text in enumerate(texts):
        ids, mask = batch['input_ids'][i], batch['attention_mask'][i]
        valid = ids[mask]
        labels = ids[1:].masked_fill(~mask[1:], -100)
        print('\nText:', repr(text))
        print('tokens:        ', tokenizer.convert_ids_to_tokens(ids.tolist()))
        print('input_ids:     ', ids.tolist())
        print('attention_mask:', mask.int().tolist())
        print('ITG labels:    ', labels.tolist())
        print(f'PASS: đầu={tokenizer.cls_token}/{valid[0].item()}, '
              f'cuối hợp lệ={tokenizer.sep_token}/{valid[-1].item()}, PAD={(~mask).sum().item()}')
    print('SEP là token hợp lệ cuối; phần tử cuối tensor có thể là PAD khi batch có câu ngắn hơn.')
    print('MAX_LENGTH tính cả CLS/SEP; truncation vẫn phải giữ SEP. Câu rỗng là [CLS, SEP].')
    print('Hiện ITC/ITM/ITG nhận cùng input_ids bắt đầu CLS; demo mặc định không thêm DEC.')
    print('ITG labels dịch một token, bỏ PAD bằng -100 và giữ SEP làm đích kết thúc câu.')


def make_objective_inputs(tokenizer, texts, max_length):
    """Demo CLS cho ITC/ITM, DEC cho ITG; labels đã shift theo engine hiện tại."""
    if tokenizer.bos_token != '[DEC]':
        raise ValueError('Cần đăng ký bos_token="[DEC]" trước khi tạo input ITG')
    if tokenizer.padding_side != 'right' or max_length < 2:
        raise ValueError('Yêu cầu right padding và max_length >= 2 (CLS + SEP)')
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                      return_tensors='pt', return_special_tokens_mask=True, return_offsets_mapping=True)
    if not batch['input_ids'][:, 0].eq(tokenizer.cls_token_id).all():
        raise ValueError('Demo yêu cầu tokenizer BERT tự chèn CLS ở đầu câu')
    decoder_ids = batch['input_ids'].clone()
    decoder_ids[:, 0] = tokenizer.bos_token_id
    labels = decoder_ids[:, 1:].masked_fill(~batch['attention_mask'][:, 1:].bool(), -100)
    return batch, decoder_ids, labels


def show_tokenization(tokenizer, texts, max_length):
    print('\n=== 1. Tokenizer gốc ===')
    print('model:', tokenizer.name_or_path, '| class:', type(tokenizer).__name__, '| fast:', tokenizer.is_fast)
    print('vocab_size:', tokenizer.vocab_size, '| len(tokenizer):', len(tokenizer))
    print('model_max_length:', tokenizer.model_max_length, '| max_length thử nghiệm:', max_length)
    print('padding_side:', tokenizer.padding_side, '| truncation_side:', tokenizer.truncation_side)
    print('vocab_size là vocab nền; len(tokenizer) tính cả token bổ sung, dùng khi resize embedding.')
    for name in ('pad', 'unk', 'cls', 'sep', 'mask', 'bos', 'eos'):
        print(f'{name:>4}: {getattr(tokenizer, name + "_token")!r:10} id={getattr(tokenizer, name + "_token_id")}')
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    print('backend model:', backend['model']['type'])
    for name in ('normalizer', 'pre_tokenizer', 'post_processor'):
        print(name + ':', json.dumps(backend[name], ensure_ascii=False))
    print('Với bert-base-uncased: lowercase/bỏ dấu có thể mất thông tin; ## là phần tiếp của WordPiece.')
    print('Tokenizer không có trọng số học được. Ít UNK không đồng nghĩa biểu diễn ngôn ngữ đã tốt.')
    for text in texts:
        full_ids = tokenizer(text, add_special_tokens=False)['input_ids']
        encoded = tokenizer(text, truncation=True, max_length=max_length)
        kept = len(encoded['input_ids']) - tokenizer.num_special_tokens_to_add(pair=False)
        print('\nText:', repr(text))
        normalizer = tokenizer.backend_tokenizer.normalizer
        print('normalize:', normalizer.normalize_str(text) if normalizer else text)
        print('pieces:', tokenizer.convert_ids_to_tokens(full_ids))
        print('content tokens:', len(full_ids), '| UNK:', full_ids.count(tokenizer.unk_token_id),
              '| bị cắt:', max(0, len(full_ids) - kept))
        print('decode:', repr(tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)))
    print('Decode không phục hồi nguyên văn chữ hoa/dấu đã bị normalizer loại bỏ.')


def show_objectives(tokenizer, texts, max_length):
    print('\n=== 2. Thử thêm DEC trên bản sao tokenizer ===')
    extended = deepcopy(tokenizer)
    print('Trước đăng ký, "[DEC]" ->', tokenizer.tokenize('[DEC]'))
    added = extended.add_special_tokens({'bos_token': '[DEC]'})
    print('Số token mới:', added, '| len:', len(tokenizer), '->', len(extended))
    print('Sau đăng ký, "[DEC]" ->', extended.tokenize('[DEC]'), '| id:', extended.bos_token_id)
    print('Đăng ký bos_token không thay post_processor BERT: tokenizer(text) vẫn chèn CLS.')
    batch, decoder_ids, labels = make_objective_inputs(extended, texts, max_length)
    print('Shape input_ids/attention_mask:', tuple(batch['input_ids'].shape))
    print('attention_mask: 1 = hợp lệ, 0 = PAD; đây chưa phải causal mask của ITG.')
    print('special_tokens_mask đánh dấu token đặc biệt do encoding chèn; không dùng nó để che CLS/SEP.')
    print('ITG input[:, :-1] dự đoán labels=input[:, 1:]; PAD label=-100, SEP vẫn được học.')
    print('Model hiện nhận cả chuỗi rồi bỏ logits cuối; không shift labels lần thứ hai.')
    for i, text in enumerate(texts):
        print('\nText:', repr(text))
        print('pos | encoder token/id | decoder token/id | attend | special | offset | next label')
        for pos, token_id in enumerate(batch['input_ids'][i].tolist()):
            decoder_id = decoder_ids[i, pos].item()
            target = labels[i, pos].item() if pos < labels.size(1) else None
            target_text = ('<bỏ logits cuối>' if target is None else '<ignore>' if target == -100
                           else f'{extended.convert_ids_to_tokens(target)}/{target}')
            print(f'{pos:3} | {extended.convert_ids_to_tokens(token_id)}/{token_id} | '
                  f'{extended.convert_ids_to_tokens(decoder_id)}/{decoder_id} | '
                  f'{batch["attention_mask"][i, pos].item()} | {batch["special_tokens_mask"][i, pos].item()} | '
                  f'{batch["offset_mapping"][i, pos].tolist()} | {target_text}')
    print('ITC đọc hidden state CLS làm text feature; ITM đọc logits từ query tokens.')
    print('ITG dùng DEC bắt đầu, SEP kết thúc. BERT không có eos_token: generation cần quy định dừng tại SEP.')
    print('DEC là tín hiệu học được, không tự chọn objective hoặc tạo causal mask.')
    return extended


def show_masks():
    print('\n=== 3. Mask Q-Former thực tế: hàng nhìn cột, 1 = được attention ===')
    print('Ví dụ 2 query và text [START, word, SEP, PAD].')
    for objective in ('itc', 'itm', 'itg'):
        mask = generate_mask_qformer(torch.tensor([[1, 1, 1, 0]]), 2, objective)[0, 0]
        print(objective.upper(), '\n         Q0 Q1 ST WD SEP PAD')
        for name, row in zip(('Q0', 'Q1', 'START', 'word', 'SEP', 'PAD'), mask.int().tolist()):
            print(f'{name:>5}:  ' + '  '.join(map(str, row)))
    print('PAD bị che ở cột (key), không nhất thiết ở hàng (query); loss phải bỏ PAD target.')
    print('ITC tách query/text; ITM hai chiều; ITG cho text nhìn query và quá khứ, query không nhìn text.')


def show_embeddings(tokenizer, extended, local_files_only, inspect_embeddings):
    cfg = AutoConfig.from_pretrained(config.BERT_MODEL_ID, local_files_only=local_files_only)
    print('\n=== 4. Embedding và quyết định đóng/mở khóa ===')
    print('Word embedding theo config:', (cfg.vocab_size, cfg.hidden_size), '| sau thêm DEC:', (len(extended), cfg.hidden_size))
    print('Position:', (cfg.max_position_embeddings, cfg.hidden_size), '| token_type:', (cfg.type_vocab_size, cfg.hidden_size))
    print('BertEmbeddings = word + position + token_type, rồi LayerNorm và dropout.')
    print('PAD không nhất thiết là vector 0 trong checkpoint; attention mask và loss mask vẫn cần thiết.')
    print('Pipeline dùng token_type mặc định 0; token_type không phải CLS/DEC. Query tokens là Parameter riêng.')
    print(f'DEC id={extended.bos_token_id}; ID >= số hàng embedding sẽ gây index out of range.')
    print('Thêm token chỉ đổi tokenizer; cần resize word embedding tới len(tokenizer).')
    print('ModelStage1 là nn.Module tùy chỉnh, không có sẵn resize_token_embeddings như BertModel.')
    print('lm_head.weight dùng chung Parameter với word_embeddings.weight: resize cần nối lại weight tie,')
    print('cập nhật kích thước output/config vocab và tạo optimizer SAU resize/mở khóa.')
    print('Hiện cả BertEmbeddings bị đóng băng; lm_head cũng bị đóng băng do weight tie.')
    print('eval() tắt dropout, không đóng băng gradient; requires_grad_(False) mới đóng băng gradient.')
    print('Mở word_embeddings cũng mở ma trận output LM; mở toàn embeddings còn mở position/token_type/LayerNorm.')
    print('DEC mới không có vector pretrained mang nghĩa DEC: có thể khởi tạo từ CLS hoặc thống kê vocab.')
    print('Giữ vector mới cố định vẫn cho tầng sau học cách dùng nó, nhưng bản thân vector không được tối ưu.')
    print('Chỉ học hàng DEC: mask gradient chưa đủ nếu AdamW vẫn áp dụng weight decay/momentum lên hàng cũ.')
    print('EMA hiện chỉ chứa itc_encoder; teacher dùng embeddings online. Cần xem lại điều này nếu mở embeddings.')
    print('Cùng vocab size nhưng khác ánh xạ token→ID vẫn làm sai pretrained embeddings.')
    if not inspect_embeddings:
        print('Thêm --inspect-embeddings để kiểm chứng bảng pretrained và resize trên model tạm trong RAM.')
        return
    bert = BertModel.from_pretrained(config.BERT_MODEL_ID, add_pooling_layer=False, local_files_only=local_files_only)
    bert.embeddings.requires_grad_(False).eval()
    print('\nKiểm tra BERT pretrained trong RAM, không ghi weights/tokenizer ra đĩa:')
    for name, parameter in bert.embeddings.named_parameters():
        print(name, tuple(parameter.shape), 'requires_grad=', parameter.requires_grad)
    old_weight = bert.embeddings.word_embeddings.weight
    old_values = old_weight.detach().clone()
    print('Norm PAD/CLS/SEP:', {name: old_weight[token_id].norm().item() for name, token_id in
                               [('PAD', tokenizer.pad_token_id), ('CLS', tokenizer.cls_token_id), ('SEP', tokenizer.sep_token_id)]})
    bert.resize_token_embeddings(len(extended))
    new_weight = bert.embeddings.word_embeddings.weight
    print('Resize:', tuple(old_values.shape), '->', tuple(new_weight.shape))
    print('Giữ nguyên hàng pretrained:', torch.equal(old_values, new_weight[:len(old_values)]))
    print('Parameter vẫn là object cũ:', old_weight is new_weight, '| requires_grad:', new_weight.requires_grad)
    print('Norm hàng DEC mới:', new_weight[extended.bos_token_id].norm().item())
    print('Nếu LM head/optimizer giữ Parameter cũ, phải cập nhật sau thao tác thay embedding.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', action='append', help='Lặp --text để xem padding giữa nhiều câu')
    parser.add_argument('--max-length', type=int, default=config.MAX_LENGTH)
    parser.add_argument('--local-files-only', action='store_true', help='Chỉ dùng model/tokenizer đã cache')
    parser.add_argument('--demo-dec', action='store_true', help='Thử thêm DEC riêng; không thuộc pipeline hiện tại')
    parser.add_argument('--inspect-embeddings', action='store_true', help='Nạp BERT pretrained để thử resize')
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error('--max-length phải >= 2')
    if args.inspect_embeddings and not args.demo_dec:
        parser.error('--inspect-embeddings cần --demo-dec để thử resize cho token mới')
    texts = args.text or ['A DOG is running.', 'The dog is playing unbelievably happily in the garden.',
                          'Một chú chó đang chạy trên cỏ.', '']
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_MODEL_ID, use_fast=config.TOKENIZER_USE_FAST,
                                              local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer has no PAD/EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    print('HIỆN TRẠNG: dataloader/engine chưa đăng ký DEC hoặc đổi CLS→DEC cho ITG.')
    print('Đây là công cụ kiểm tra độc lập; chưa được nối vào training.')
    show_tokenization(tokenizer, texts, args.max_length)
    show_vlm_batch(tokenizer, texts, args.max_length)
    show_masks()
    if args.demo_dec:
        extended = show_objectives(tokenizer, texts, args.max_length)
        show_embeddings(tokenizer, extended, args.local_files_only, args.inspect_embeddings)


if __name__ == '__main__':
    main()
