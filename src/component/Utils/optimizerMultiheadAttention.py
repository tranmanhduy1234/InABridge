import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.attention import SDPBackend

class OptimizedFlashMHA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, bias=True, dropout_p=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_head = num_heads
        self.head_dim = embed_dim // num_heads
        self.drop_out_p = dropout_p
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, val=0.0)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, val=0.0)
            
    # kv_cache: batch_size, numhead, seq_len, d_model
    def forward(self, query, key, value, key_padding_mask=None, is_causal=False,
                use_cache=False, kv_cache=None, attn_mask=None):
        B, T, D = query.shape
        src_len = key.size(1)
        
        # ======= Self-Attention =======
        if query is key and key is value:
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            qkv = qkv.view(B, T, 3, self.num_head, self.head_dim)
            q, k, v = qkv.unbind(dim=2) # batch_size, seqlen, num_head, head_dim
            
            q = q.transpose(1, 2).contiguous() # [batch_size, numheam, seqlen, head_dim]
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            if use_cache:
                if kv_cache is not None:
                    k_past, v_past = kv_cache
                    k = torch.cat((k_past, k), dim=2)
                    v = torch.cat((v_past, v), dim=2)
                kv_cache = (k, v)
        # ======= Cross-Attention =======
        else:
            w = self.in_proj_weight
            b = self.in_proj_bias
            
            q = F.linear(query, w[:D], b[:D] if b is not None else None)
            q = q.view(B, T, self.num_head, self.head_dim)
            q = q.transpose(1, 2).contiguous()
            
            if kv_cache is not None and use_cache:
                k, v = kv_cache
            else:                
                k = F.linear(key, w[D:2*D], b[D:2*D] if b is not None else None)
                v = F.linear(value, w[2*D:], b[2*D:] if b is not None else None)
                
                k = k.view(B, src_len, self.num_head, self.head_dim)
                v = v.view(B, src_len, self.num_head, self.head_dim)
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                
                if use_cache:
                    kv_cache = (k, v)
                    
        src_len = k.size(2)
        
        # Có 1 cái hay đó là khi inference, đối với các trường hợp có promt, trong lần đầu chạy ta vẫn sẽ phải có causal mask, kể cả có cache hay không
        # Các lần sau đó có cache, thì sẽ ko cần causal, bởi ta chỉ duy trì đúng 1 phần tử query duy nhất.
        if attn_mask is None:
            attn_mask = self._create_mask(T=T, src_len=src_len, 
                                          is_causal=is_causal, 
                                          key_padding_mask=key_padding_mask, 
                                          device=query.device)
        
        try:
            with torch.nn.attention.sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.drop_out_p if self.training else 0.0
                )
        except Exception:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.drop_out_p if self.training else 0.0
            )
            
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.embed_dim)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, kv_cache
    
    def _create_mask(self, T, src_len, is_causal, key_padding_mask, device):
        if is_causal and key_padding_mask is None:
            causal_mask = torch.tril(
                torch.ones(T, src_len, dtype=torch.bool, device=device)
            )
            return causal_mask.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, T, src_len)
        elif is_causal and key_padding_mask is not None:
            causal_mask = torch.tril(
                torch.ones(T, src_len, dtype=torch.bool, device=device)
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, src_len)
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, src_len)
            combined_mask = causal_mask & padding_mask
            return combined_mask.contiguous()
        elif not is_causal and key_padding_mask is not None:
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).contiguous()  # (B, 1, 1, src_len)
            return padding_mask
        return None
    
if __name__=="__main__":
    import torch
    from torch.nn.attention import SDPBackend

    # Giả sử lớp OptimizedFlashMHA của bạn đã được định nghĩa ở trên
    # từ mô phỏng thiết bị phần cứng (SDPA Efficient Attention tối ưu nhất trên CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy demo trên thiết bị: {device}")
    if device.type == "cpu":
        print("⚠️ Cảnh báo: SDPBackend.EFFICIENT_ATTENTION tối ưu tốt nhất trên GPU CUDA.")

    # Khởi tạo siêu tham số (Hyperparameters)
    B = 2          # Batch size
    Embed_dim = 512
    Num_heads = 8

    # Khởi tạo mô hình
    mha = OptimizedFlashMHA(embed_dim=Embed_dim, num_heads=Num_heads, bias=True, dropout_p=0.1).to(device)
    mha.eval()  # Chuyển sang chế độ eval để loại bỏ dropout ngẫu nhiên khi chạy demo

    # =====================================================================
    # TRƯỜNG HỢP 1: Self-Attention thông thường (Không dùng Cache, có Causal Mask)
    # Phù hợp cho giai đoạn tiền xử lý câu Prompt ban đầu (Prefill Phase)
    # =====================================================================
    print("\n--- Trường hợp 1: Self-Attention với Causal Mask (Giai đoạn Prefill) ---")
    T_prompt = 4  # Chiều dài chuỗi prompt ban đầu
    X_prompt = torch.randn(B, T_prompt, Embed_dim, device=device)

    # Định nghĩa key_padding_mask: True là token thật, False là padding
    # Giả sử batch 1 có 4 token hợp lệ, batch 2 chỉ có 3 token hợp lệ (1 token padding ở cuối)
    key_padding_mask = torch.tensor([
        [True, True, True, True],
        [True, True, True, False]
    ], dtype=torch.bool, device=device)

    # Chạy mô hình (query = key = value)
    output_prompt, kv_cache = mha(
        query=X_prompt, key=X_prompt, value=X_prompt,
        key_padding_mask=key_padding_mask,
        is_causal=True,
        use_cache=True, # Bật chế độ lưu cache để chuẩn bị cho bước sau
        kv_cache=None   # Lần đầu tiên chạy nên chưa có cache quá khứ
    )

    print(f"Kích thước Output: {output_prompt.shape}") # mong đợi: (2, 4, 512)
    print(f"Kích thước K-Cache lưu lại: {kv_cache[0].shape}") # mong đợi: (2, 8, 4, 64) -> (B, nh, T, hd)


    # =====================================================================
    # TRƯỜNG HỢP 2: Từng bước sinh Token kế tiếp sử dụng KV Cache
    # Mô phỏng quá trình sinh chữ tuần tự khi Inference (Incremental Decoding)
    # =====================================================================
    print("\n--- Trường hợp 2: Sinh Token kế tiếp tuần tự sử dụng KV Cache ---")

    # Giả định chúng ta sinh tiếp 2 token mới độc lập, lần lượt từng bước một
    T_generate_steps = 2

    for step in range(T_generate_steps):
        # Tại mỗi bước sinh, Query chỉ nhận duy nhất 1 token mới vừa được sinh ra ở bước trước
        # Chiều dài sequence length lúc này luôn luôn T = 1
        X_new_token = torch.randn(B, 1, Embed_dim, device=device)
        
        # Khi đã dùng KV Cache, token hiện tại chỉ cần nhìn lại toàn bộ quá khứ và chính nó.
        # Vì thế ta không cần key_padding_mask phức tạp hay causal_mask nữa (hệ thống tự động hiểu nhờ cache tăng dần)
        output_step, kv_cache = mha(
            query=X_new_token, 
            key=X_new_token, 
            value=X_new_token,
            key_padding_mask=None, 
            is_causal=False, # Không cần causal mask vì T=1
            use_cache=True,
            kv_cache=kv_cache # Truyền bộ nhớ cache từ bước trước vào đây
        )
        
        print(f"Bước {step + 1}:")
        print(f"  -> Kích thước mã nhúng mới đưa vào: {X_new_token.shape}") # (2, 1, 512)
        print(f"  -> Kích thước Output đầu ra: {output_step.shape}")       # (2, 1, 512)
        print(f"  -> Kích thước K-Cache hiện tại: {kv_cache[0].shape}")    # Tăng dần qua từng bước


    # =====================================================================
    # TRƯỜNG HỢP 3: Cross-Attention 
    # Mô phỏng Q-Former (hoặc Decoder) tương tác với Image Encoder (hoặc Encoder)
    # =====================================================================
    print("\n--- Trường hợp 3: Cross-Attention (Decoder attending to Encoder) ---")

    T_decoder = 3  # Giả sử Decoder / Q-Former có 3 học phần truy vấn (queries)
    T_encoder = 6  # Giả sử Image Encoder / Văn bản nguồn xuất ra 6 token đặc trưng

    # Tạo dữ liệu giả lập riêng biệt cho các thành phần
    query_features = torch.randn(B, T_decoder, Embed_dim, device=device)
    encoder_features = torch.randn(B, T_encoder, Embed_dim, device=device)

    # Thường trong Cross-Attention, ta không dùng Causal Mask (is_causal=False) 
    # vì Decoder cần tương tác tự do với mọi vùng đặc trưng thu được từ Encoder
    output_cross, _ = mha(
        query=query_features, 
        key=encoder_features, 
        value=encoder_features,
        key_padding_mask=None, 
        is_causal=False,
        use_cache=False
    )

    print(f"Kích thước đặc trưng Query đầu vào: {query_features.shape}")   # (2, 3, 512)
    print(f"Kích thước đặc trưng Encoder đầu vào: {encoder_features.shape}") # (2, 6, 512)
    print(f"Kích thước Output Cross-Attention: {output_cross.shape}")     # (2, 3, 512)