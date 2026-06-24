import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenWithQueriesModule(nn.Module):
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", device="cpu", dtype=torch.bfloat16, num_queries=64):
        super().__init__()
        self.device = device
        self.num_queries = num_queries
        
        # 1. Tải phần lõi của mô hình CausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # 2. TÁCH BIỆT: Trích xuất tầng Embedding của mô hình để xử lý thủ công
        self.embeddings = self.model.get_input_embeddings()
        self.hidden_size = self.model.config.hidden_size
        
        # 3. QUERIES: Khởi tạo 64 learnable queries dưới dạng nn.Parameter (sẽ được cập nhật khi train)
        self.queries = nn.Parameter(
            torch.randn(self.num_queries, self.hidden_size, dtype=dtype, device=self.device)
        )
        print(f"[✓] Khởi tạo mô hình thành công. Đã tách biệt Embedding và nhúng {num_queries} Queries.\n")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Pipeline dành cho TRAINING / VALIDATION.
        Nhận vào tensor đã tokenize, tự chuyển qua Embedding và nối với 64 queries.
        """
        batch_size = input_ids.size(0)
        
        # Bước tách biệt: Thực hiện embedding lookup cho phần Instruct text độc lập
        instruct_embeds = self.embeddings(input_ids) # Shape: (batch, seq_len, hidden_size)
        
        # Mở rộng bộ 64 queries theo kích thước của batch
        query_embeds = self.queries.unsqueeze(0).expand(batch_size, -1, -1) # Shape: (batch, 64, hidden_size)
        
        # NỐI VECTORS: Ghép 64 queries vào TRƯỚC chuỗi Instruct embeddings
        inputs_embeds = torch.cat([query_embeds, instruct_embeds], dim=1) # Shape: (batch, 64 + seq_len, hidden_size)
        
        # Cập nhật attention_mask: Cho phép mô hình chú ý (attend) vào 64 queries ở đầu
        if attention_mask is not None:
            query_mask = torch.ones(batch_size, self.num_queries, dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat([query_mask, attention_mask], dim=1)
            
        # Cập nhật labels: Đặt nhãn cho 64 slots của queries là -100 để hàm Loss bỏ qua không tính toán
        if labels is not None:
            query_labels = torch.full((batch_size, self.num_queries), -100, dtype=labels.dtype, device=self.device)
            labels = torch.cat([query_labels, labels], dim=1)
            
        # CHẠY LÕI MÔ HÌNH: Truyền trực tiếp 'inputs_embeds' thay vì 'input_ids'
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs  # Trả về object chứa .loss và .logits

    def generate_from_embeddings(self, input_ids, attention_mask=None, max_new_tokens=100, do_sample=False):
        """
        Pipeline dành cho INFERENCE / SUY LUẬN.
        Hàm generate của Hugging Face hỗ trợ nhận trực tiếp inputs_embeds đầu vào.
        """
        batch_size = input_ids.size(0)
        
        # Chuẩn bị embedding tổng hợp (Queries + Instruct) tương tự như hàm forward
        instruct_embeds = self.embeddings(input_ids)
        query_embeds = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([query_embeds, instruct_embeds], dim=1)
        
        if attention_mask is not None:
            query_mask = torch.ones(batch_size, self.num_queries, dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat([query_mask, attention_mask], dim=1)
            
        with torch.no_grad():
            # Khi truyền inputs_embeds, hàm generate chỉ sinh ra và trả về những Token MỚI (new tokens)
            generated_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.eos_token_id
            )
        return generated_ids


# ========================================================
# KHỞI CHẠY PIPELINE (TOKENIZER ĐẰNG NGOÀI)
# ========================================================
if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    device = "cpu" # Chuyển thành "cuda" nếu bạn chạy trên GPU
    
    # 1. KHỞI TẠO TOKENIZER Ở NGOÀI MÔ HÌNH
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. KHỞI TẠO MÔ HÌNH LÕI (Chứa 64 Queries & Tầng Embedding tách biệt)
    custom_model = QwenWithQueriesModule(model_id=model_id, device=device, dtype=torch.bfloat16, num_queries=64)

    # ----------------------------------------------------
    # DEMO 1: PIPELINE INFERENCE 
    # ----------------------------------------------------
    print("--- DEMO INFERENCE ---")
    prompt = "2 mũ 10 bằng bao nhiêu\nSuy luận:"
    
    # Tokenize ở ngoài pipeline mô hình
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    # Gọi hàm sinh token từ tầng embedding đã trộn queries
    new_token_ids = custom_model.generate_from_embeddings(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50
    )
    # Decode kết quả ở ngoài
    output_text = tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
    
    print(f"[✓] Thời gian xử lý: {time.time() - start_time:.2f} giây.")
    print(f"Kết quả suy luận (sau 64 queries):\n{output_text}")
    print("="*40 + "\n")

    # ----------------------------------------------------
    # DEMO 2: PIPELINE TRAINING LOOP
    # ----------------------------------------------------
    print("--- DEMO TRAINING LOOP ---")
    custom_model.train()
    
    text_input = "Học máy là gì?"
    text_target = "Học máy là một nhánh của trí tuệ nhân tạo."
    
    # Tokenize dữ liệu thô ở ngoài
    full_text = text_input + " " + text_target
    inputs_train = tokenizer(full_text, return_tensors="pt").to(device)
    
    input_ids = inputs_train["input_ids"]
    attention_mask = inputs_train["attention_mask"]
    labels = input_ids.clone() # Causal LM tiêu chuẩn

    # Chạy bước Forward pass qua mô hình
    outputs = custom_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"Loss thu được từ bước forward: {outputs.loss.item():.4f}")
    print("[✓] Thiết lập kiến trúc Tách biệt Embedding & Tokenizer thành công!")