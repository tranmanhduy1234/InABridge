import torch
from transformers import AutoTokenizer, AutoModel

# Định danh mô hình DeBERTa-v3 Base chuẩn từ Microsoft
model_id = "microsoft/deberta-v3-base"

print(f"Đang tải Tokenizer và Model {model_id} từ Hugging Face...")

# Tải bộ mã hóa ngôn ngữ (Tokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Tải kiến trúc mô hình chính
model = AutoModel.from_pretrained(model_id)
model.eval()  # Chuyển sang chế độ eval phục vụ inference/bridge

print("Tải mô hình DeBERTa-v3 thành công!")

# --- Thử nghiệm chạy với một BATCH văn bản mẫu ---
# Bạn có thể truyền vào một câu hoặc một list nhiều câu (Batch)
batch_texts = [
    "Đây là câu văn mẫu thứ nhất để kiểm tra token đầu vào.",
    "Hệ thống InstructBLIP đang kết nối SigLIP2 với Qwen3.5 thông qua DeBERTa."
]

# Tiến hành tokenize batch văn bản
# padding=True và truncation=True giúp các câu trong batch có độ dài bằng nhau
inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

print("\n--- Kết quả Tokenize ---")
print("Kích thước Input IDs:", inputs['input_ids'].shape)
# Dạng: [Batch_Size, Sequence_Length]

# Đưa qua mô hình DeBERTa-v3 để lấy đặc trưng (Embeddings)
with torch.no_grad():
    outputs = model(**inputs)

# Lấy trạng thái ẩn cuối cùng (Last Hidden State)
last_hidden_state = outputs.last_hidden_state

print("\n--- Kết quả trích xuất đặc trưng (Embeddings) ---")
print("Kích thước đặc trưng đầu ra (Last Hidden State):", last_hidden_state.shape)
# Kết quả sẽ có dạng: torch.Size([2, độ_dài_chuỗi, 768])
# Trong đó 768 chính là kích thước ẩn (Hidden Size) mặc định của DeBERTa-v3 Base.