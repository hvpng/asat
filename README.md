# ASAT — Attribution-Stable Adversarial Training for NLP

> **Research project** | Target: NAACL 2027 / EMNLP Findings 2027

## Vấn đề

Adversarial Training (AT) hiện tại (FreeLB, InfoBERT) giúp model giữ stable **classification label** dưới adversarial perturbation. Nhưng **attribution maps** (explanation của model) vẫn có thể sụp đổ hoàn toàn dù nhãn không đổi — *explanation-only attack*.

## Đề xuất

ASAT thêm một **Attribution Alignment Loss** vào training objective của FreeLB:

```
L_total = L_cls + λ₁·L_adv + λ₂·L_align
```

Trong đó `L_align = MSE(ā(X,y), ā(X+δ,y))` ép attribution map của clean và adversarial input phải giống nhau, được tính bằng **Gradient×Input** (proxy nhanh, differentiable) trong training và **Integrated Gradients** (axiomatic) trong evaluation.

## Cấu trúc repo

```
asat/
├── configs/          # YAML hyperparameters
├── data/             # Dataset loading (SST-2, IMDB, Yelp)
├── models/           # Model wrappers
├── attribution/      # Grad×Input (train) và IG (eval)
├── training/         # train_standard.py, freelb.py, asat.py
├── evaluation/       # metrics.py, attack_eval.py, bootstrap.py
├── experiments/      # Shell scripts chạy all configs
├── results/          # CSV/JSON output
├── notebooks/        # EDA và analysis
└── checkpoints/      # Model weights (gitignored)
```

## Quick start

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Copy env template
cp .env.example .env  # điền WANDB_API_KEY

# 3. Chạy baseline (tuần 2)
python training/train_standard.py --dataset sst2 --seed 42

# 4. Chạy FreeLB (tuần 4)
python training/freelb.py --dataset sst2 --seed 42

# 5. Chạy ASAT (tuần 7)
python training/asat.py --dataset sst2 --seed 42
```

## Baselines

| Method | Space | XAI alignment | Adv. attr. alignment | Model-agnostic |
|--------|-------|---------------|----------------------|----------------|
| No defense | — | ✗ | ✗ | ✓ |
| FreeLB | Embedding | ✗ | ✗ | ✓ |
| FLAT | Discrete | ✓ | ✓ (discrete) | ✗ |
| REGEX | Embedding | ✓ (clean only) | ✗ | ✓ |
| **ASAT** | **Embedding** | **✓** | **✓** | **✓** |

## Metrics

- **Clean Accuracy ↑** — accuracy trên test set không bị attack
- **AUA ↑** — accuracy sau TextFooler / BERT-Attack
- **S_adv ↓** — `1 - Spearman(IG_clean, IG_attacked)`, chỉ trên samples có prediction không thay đổi

## Timeline

- Tuần 1–4: Setup + Baseline + FreeLB reproduce
- Tuần 5–7: Implement ASAT (L_align)
- Tuần 8–10: Evaluation đầy đủ (3 seeds, 2 attacks)
- Tuần 11–18: Mở rộng datasets + ablation
- Tuần 19–28: Viết paper + submit ARR