import torch
from torch import nn
import torch.nn.functional as F
class MoCoQueue(nn.Module):
    def __init__(self, capacity: int, num_queries: int, dim: int):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("image", torch.zeros(capacity, num_queries, dim))
        self.register_buffer("text", torch.zeros(capacity, dim))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, images, texts):
        b = images.size(0)
        n = min(b, self.capacity)

        start = (self.ptr.item() + b - n) % self.capacity
        idx = (torch.arange(n, device=images.device) + start) % self.capacity

        self.image[idx] = images[-n:].detach()
        self.text[idx] = texts[-n:].detach()
        self.ptr.fill_((self.ptr.item() + b) % self.capacity)
        self.count.fill_(min(self.capacity, self.count.item() + b))

    def get(self):
        n = self.count.item()
        if n < self.capacity:
            return self.image[:n], self.text[:n]
        idx = (
            torch.arange(self.capacity, device=self.image.device)
            + self.ptr
        ) % self.capacity
        return self.image[idx], self.text[idx]

def main():
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)

    B = 4
    Q = 128
    D = 256
    CAPACITY = 10
    temperature = 0.07

    device = "cuda" if torch.cuda.is_available() else "cpu"
    queue = MoCoQueue(CAPACITY, Q, D).to(device)

    # =========================================================
    # ITERATION 1
    # Chỉ tạo feature rồi enqueue để fill queue
    # =========================================================

    image_1 = F.normalize(
        torch.randn(B, Q, D, device=device),
        dim=-1
    )

    text_1 = F.normalize(
        torch.randn(B, D, device=device),
        dim=-1
    )

    print("=== ITERATION 1 ===")
    print("Current image:", image_1.shape)
    print("Current text :", text_1.shape)

    image_q, text_q = queue.get()

    print("Queue before enqueue")
    print("image queue:", image_q.shape)
    print("text queue :", text_q.shape)
    print("queue len  :", len(text_q))

    # Fill queue
    queue.enqueue(image_1, text_1)

    image_q, text_q = queue.get()

    print("\nQueue after enqueue")
    print("image queue:", image_q.shape)
    print("text queue :", text_q.shape)
    print("queue len  :", len(text_q))
    print("ptr        :", queue.ptr.item())

    # =========================================================
    # ITERATION 2
    # Đây mới là lượt thấy rõ MoCo queue hoạt động
    # =========================================================

    image_2 = F.normalize(
        torch.randn(B, Q, D, device=device),
        dim=-1
    )

    text_2 = F.normalize(
        torch.randn(B, D, device=device),
        dim=-1
    )

    image_q, text_q = queue.get()

    print("\n\n=== ITERATION 2 ===")
    print("Current image:", image_2.shape)
    print("Current text :", text_2.shape)

    print("\nExisting queue")
    print("image queue:", image_q.shape)
    print("text queue :", text_q.shape)
    print("queue len  :", len(text_q))

    # ---------------------------------------------------------
    # IMAGE -> TEXT
    # ---------------------------------------------------------

    # Current batch text + old queue text
    all_text = torch.cat(
        [text_2, text_q],
        dim=0
    )

    print("\n--- Image -> Text ---")
    print("In-batch text :", text_2.shape)
    print("Queue text    :", text_q.shape)
    print("All text      :", all_text.shape)

    # image_2 : [B, Q, D]
    # all_text: [B + L, D]
    #
    # -> [B, Q, B + L]
    sim_i2t = torch.einsum(
        "bqd,kd->bqk",
        image_2,
        all_text
    )

    print("Similarity before max:", sim_i2t.shape)

    # Max over 128 queries
    sim_i2t = sim_i2t.max(dim=1).values

    # [B, B + L]
    logits_i2t = sim_i2t / temperature

    print("Logits i2t:", logits_i2t.shape)

    print(
        f"Expected logits width = "
        f"in_batch({B}) + queue({len(text_q)}) "
        f"= {B + len(text_q)}"
    )

    # ---------------------------------------------------------
    # TEXT -> IMAGE
    # ---------------------------------------------------------

    all_image = torch.cat(
        [image_2, image_q],
        dim=0
    )

    print("\n--- Text -> Image ---")
    print("In-batch image:", image_2.shape)
    print("Queue image   :", image_q.shape)
    print("All image     :", all_image.shape)

    # text_2   : [B, D]
    # all_image: [B + L, Q, D]
    #
    # -> [B, B + L, Q]
    sim_t2i = torch.einsum(
        "bd,kqd->bkq",
        text_2,
        all_image
    )

    print("Similarity before max:", sim_t2i.shape)

    sim_t2i = sim_t2i.max(dim=-1).values

    # [B, B + L]
    logits_t2i = sim_t2i / temperature

    print("Logits t2i:", logits_t2i.shape)

    print(
        f"Expected logits width = "
        f"in_batch({B}) + queue({len(image_q)}) "
        f"= {B + len(image_q)}"
    )

    # ---------------------------------------------------------
    # Positive targets
    # ---------------------------------------------------------

    # all_text / all_image:
    #
    # [current batch | queue]
    #
    # nên positive của sample i vẫn nằm tại index i
    targets = torch.arange(B, device=device)

    loss_i2t = F.cross_entropy(logits_i2t, targets)
    loss_t2i = F.cross_entropy(logits_t2i, targets)

    loss = 0.5 * (loss_i2t + loss_t2i)

    print("\n--- Loss ---")
    print("loss_i2t:", loss_i2t.item())
    print("loss_t2i:", loss_t2i.item())
    print("loss     :", loss.item())

    # Cuối iteration 2 mới enqueue batch mới
    queue.enqueue(image_2, text_2)

    image_q, text_q = queue.get()

    print("\n--- Queue after iteration 2 ---")
    print("image queue:", image_q.shape)
    print("text queue :", text_q.shape)
    print("queue len  :", len(text_q))
    print("ptr        :", queue.ptr.item())

if __name__ == "__main__":
    main()
