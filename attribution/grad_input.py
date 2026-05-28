# attribution/grad_input.py
# Gradient × Input (GI) — dùng cho TRAINING (asat.py, tuần 9+).
#
# Theo đề cương:
#   - GI là proxy bậc một, differentiable → có thể backward qua L_align.
#   - Yêu cầu create_graph=True để PyTorch giữ đồ thị tính toán cho
#     double backpropagation (đạo hàm bậc hai qua L_align).
#   - Tính đạo hàm theo cùng ground-truth label y (target-consistent).
#   - Công thức: GI_i = ∂score_y/∂z_i ⊙ z_i  (Hadamard, per token per dim)
#
# API:
#   from attribution.grad_input import compute_grad_input
#   attr = compute_grad_input(model, embeddings, attention_mask, label)
#   # attr: Tensor [batch, seq_len, hidden_dim], có gradient → dùng được trong loss

import torch


def compute_grad_input(model, embeddings, attention_mask, label):
    """
    Tính Gradient × Input attribution trong training loop.

    Args:
        model         : HuggingFace model (train mode, requires_grad=True)
        embeddings    : Tensor [batch, seq_len, hidden_dim], inputs_embeds,
                        phải có requires_grad=True để lấy gradient
        attention_mask: Tensor [batch, seq_len]
        label         : Tensor [batch] — ground-truth labels (target-consistent)

    Returns:
        gi : Tensor [batch, seq_len, hidden_dim]
             GI_i = grad_i ⊙ embedding_i
             Còn trong computation graph (create_graph=True) →
             L_align.backward() sẽ tính được đạo hàm bậc hai.

    Lưu ý khi dùng:
        embeddings phải được tạo với requires_grad=True trước khi gọi hàm này.
        Không gọi model.zero_grad() giữa compute_grad_input và loss.backward().
    """
    assert embeddings.requires_grad, (
        "embeddings phải có requires_grad=True. "
        "Tạo bằng: emb = embedding_layer(input_ids).requires_grad_(True)"
    )

    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits  = outputs.logits  # [batch, num_labels]

    # Score của đúng class y cho từng sample trong batch
    # gather: [batch, 1] → squeeze → [batch]
    scores = logits.gather(dim=1, index=label.unsqueeze(1)).squeeze(1)
    score_sum = scores.sum()  # scalar để backward

    # Gradient của score_y theo embeddings
    # create_graph=True: giữ đồ thị cho double backprop qua L_align
    grads = torch.autograd.grad(
        outputs     = score_sum,
        inputs      = embeddings,
        create_graph= True,   # BẮT BUỘC cho L_align.backward()
        retain_graph= True,   # training loop cần backward thêm lần nữa
    )[0]  # [batch, seq_len, hidden_dim]

    gi = grads * embeddings  # Hadamard: ∂score/∂z ⊙ z
    return gi