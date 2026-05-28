# attribution/ig.py
# Integrated Gradients (IG) — dùng cho EVALUATION.
#
# Theo đề cương:
#   - IG là metric chính thức để tính S_adv (50 steps, axiomatic).
#   - Tính đạo hàm theo cùng ground-truth label y (target-consistent).
#   - Baseline = zero embedding (chuẩn NLP).
#   - KHÔNG dùng trong training vì quá chậm (50 forward pass/sample).
#     → Training dùng Grad×Input (attribution/grad_input.py).
#
# API:
#   from attribution.ig import compute_ig
#   tokens, attr = compute_ig(model, tokenizer, text, label, device)

import torch
import numpy as np
from captum.attr import LayerIntegratedGradients


def _get_embedding_layer(model):
    if hasattr(model, "bert"):
        return model.bert.embeddings.word_embeddings
    elif hasattr(model, "roberta"):
        return model.roberta.embeddings.word_embeddings
    else:
        raise ValueError(f"Unsupported architecture: {type(model)}")


def compute_ig(model, tokenizer, text, label, device, max_length=128, n_steps=50):
    """
    Tính token-level attribution bằng Integrated Gradients qua captum.

    Args:
        model      : HuggingFace sequence classification model (eval mode)
        tokenizer  : tương ứng
        text       : chuỗi văn bản gốc (str)
        label      : ground-truth label (int) — target-consistent, không đổi
                     dù clean hay attacked
        device     : torch.device
        max_length : max token length
        n_steps    : số bước Riemann (đề cương: 50)

    Returns:
        tokens      : list[str] — tất cả tokens kể cả [CLS]/[SEP]/[PAD]
        attributions: np.ndarray shape [seq_len] — L2 norm qua hidden dim,
                      non-negative, chưa normalize (để metrics.py tự quyết định)
    """
    model.eval()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).to(device)

    input_ids      = enc["input_ids"]       # [1, seq_len]
    attention_mask = enc["attention_mask"]  # [1, seq_len]
    baseline_ids   = torch.zeros_like(input_ids)  # all-PAD baseline

    def forward_func(ids):
        return model(input_ids=ids, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(forward_func, _get_embedding_layer(model))

    attributions, _ = lig.attribute(
        inputs    = input_ids,
        baselines = baseline_ids,
        target    = label,
        n_steps   = n_steps,
        return_convergence_delta=True,
    )
    # attributions: [1, seq_len, hidden_dim]
    attr_norm = attributions.squeeze(0).norm(dim=-1).detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())
    return tokens, attr_norm