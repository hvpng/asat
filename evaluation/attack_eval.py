# evaluation/attack_eval.py
# Orchestrator cho tuần 5–6 (và tái sử dụng mọi tuần sau).
#
# Nhiệm vụ:
#   1. Load model + dataset
#   2. Chạy TextFooler hoặc BERT-Attack qua TextAttack
#   3. Tính IG cho mỗi cặp (clean, attacked) qua attribution/ig.py
#   4. Gọi metrics.py → AUA, S_adv
#   5. Lưu JSON kết quả + heatmap PNG (nếu --visualize)
#
# Usage:
#   python evaluation/attack_eval.py \
#       --checkpoint checkpoints/standard \
#       --dataset sst2 \
#       --attack textfooler \
#       --n_samples 500 \
#       --visualize 10

import argparse
import os
import sys
import json
import torch
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from attribution.ig import compute_ig
from evaluation.metrics import compute_aua, compute_s_adv


# ── Dataset config ────────────────────────────────────────────────────────

DATASET_MAP = {
    "sst2": ("glue",          "sst2", "validation", "sentence", "label"),
    "imdb": ("imdb",          None,   "test",        "text",     "label"),
    "yelp": ("yelp_polarity", None,   "test",        "text",     "label"),
}


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset",    type=str, default="sst2",
                   choices=["sst2", "imdb", "yelp"])
    p.add_argument("--attack",     type=str, default="textfooler",
                   choices=["textfooler", "bertattack"])
    p.add_argument("--n_samples",  type=int, default=500,
                   help="Số mẫu chạy attack")
    p.add_argument("--ig_steps",   type=int, default=50)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--visualize",  type=int, default=0,
                   help="Số mẫu vẽ heatmap IG (0 = không vẽ). Tuần 5-6: 10")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


# ── TextAttack runner ─────────────────────────────────────────────────────

def run_attack(attack_name, model, tokenizer, dataset_name, n_samples, seed):
    """
    Chạy attack qua TextAttack.
    Trả về list of dict:
        { original_text, attacked_text, label, pred_flipped }
    """
    import nltk
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    from textattack import Attacker, AttackArgs
    from textattack.datasets import HuggingFaceDataset
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from textattack.attack_results import SuccessfulAttackResult
    from datasets import load_dataset as hf_load

    hf_name, hf_config, split, text_col, label_col = DATASET_MAP[dataset_name]

    # Load + shuffle trước bằng datasets (HuggingFaceDataset không nhận seed)
    raw = hf_load(hf_name, hf_config) if hf_config else hf_load(hf_name)
    ds  = raw[split].shuffle(seed=seed)

    # Wrap vào TextAttack dataset
    # dataset_columns: ([list of input cols], label_col)
    ta_dataset = HuggingFaceDataset(ds, dataset_columns=([text_col], label_col))

    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    if attack_name == "textfooler":
        from textattack.attack_recipes import TextFoolerJin2019
        attack = TextFoolerJin2019.build(wrapper)
    elif attack_name == "bertattack":
        from textattack.attack_recipes import BERTAttackLi2020
        attack = BERTAttackLi2020.build(wrapper)

    attacker = Attacker(attack, ta_dataset, AttackArgs(
        num_examples=n_samples, random_seed=seed,
        disable_stdout=True, silent=True,
    ))
    raw_results = attacker.attack_dataset()

    records = []
    for res in raw_results:
        orig    = res.original_result.attacked_text.text
        label   = int(res.original_result.ground_truth_output)
        flipped = isinstance(res, SuccessfulAttackResult)
        attacked = res.perturbed_result.attacked_text.text if flipped else orig
        records.append({
            "original_text": orig,
            "attacked_text": attacked,
            "label":         label,
            "pred_flipped":  flipped,
        })
    return records


# ── Heatmap (chỉ gọi khi --visualize > 0) ────────────────────────────────

def save_heatmap(tokens_c, attr_c, tokens_a, attr_a,
                 label, pred_flipped, rho, idx, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig, axes = plt.subplots(2, 1, figsize=(14, 3))

    def _draw(ax, tokens, attr, row_label, color):
        non_pad = [i for i, t in enumerate(tokens) if t not in ("[PAD]", "<pad>")]
        toks = [tokens[i] for i in non_pad]
        vals = attr[non_pad]
        vmax = max(np.abs(vals).max(), 1e-8)
        norm = vals / vmax
        ax.set_xlim(0, len(toks))
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_ylabel(row_label, rotation=0, labelpad=45, va="center", fontsize=8)
        for i, (tok, v) in enumerate(zip(toks, norm)):
            alpha = min(abs(v), 1.0) * 0.85
            bg = (*mpl.colors.to_rgb(color), alpha)
            ax.add_patch(plt.Rectangle([i, 0.1], 0.95, 0.8, color=bg, zorder=1))
            ax.text(i + 0.47, 0.5,
                    tok.replace("##", "").replace("[CLS]", "▶").replace("[SEP]", "◀"),
                    ha="center", va="center", fontsize=6.5, zorder=2)

    _draw(axes[0], tokens_c, attr_c, "CLEAN",    "#1a73e8")
    _draw(axes[1], tokens_a, attr_a, "ATTACKED", "#e53935")

    status = "FLIPPED ⚠" if pred_flipped else f"stable | ρ={rho:.3f}"
    fig.suptitle(f"[{idx}] label={label} | {status}", fontsize=9, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(out_dir, f"sample_{idx:02d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Attack     : {args.attack} | n_samples={args.n_samples}")

    model     = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model.eval()

    # ── 1. Chạy attack ────────────────────────────────────────────────────
    print(f"\nRunning {args.attack}...")
    records = run_attack(
        args.attack, model, tokenizer,
        args.dataset, args.n_samples, args.seed,
    )

    aua, n_total, n_flipped = compute_aua(records)
    print(f"  n_total   : {n_total}")
    print(f"  n_flipped : {n_flipped}  (prediction changed)")
    print(f"  AUA       : {aua:.4f}")

    # ── 2. Tính IG cho từng cặp → S_adv ──────────────────────────────────
    print(f"\nComputing IG ({args.ig_steps} steps) for {n_total} pairs...")
    attr_pairs   = []
    heatmap_data = []

    for i, rec in enumerate(records):
        if i % 50 == 0:
            print(f"  {i}/{n_total}", flush=True)

        tokens_c, attr_c = compute_ig(
            model, tokenizer, rec["original_text"],
            rec["label"], device, args.max_length, args.ig_steps,
        )
        tokens_a, attr_a = compute_ig(
            model, tokenizer, rec["attacked_text"],
            rec["label"], device, args.max_length, args.ig_steps,
        )

        attr_pairs.append({
            "attr_clean":    attr_c,
            "attr_attacked": attr_a,
            "pred_flipped":  rec["pred_flipped"],
        })

        if args.visualize > 0 and i < args.visualize:
            heatmap_data.append((tokens_c, attr_c, tokens_a, attr_a, rec))

    s_adv, n_stable, rho_list = compute_s_adv(attr_pairs)
    print(f"\n  S_adv     : {s_adv:.4f}  (n_stable={n_stable})")

    # ── 3. Lưu kết quả JSON ───────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_tag = os.path.basename(args.checkpoint.rstrip("/"))
    out_path = os.path.join(
        args.output_dir,
        f"{ckpt_tag}_{args.dataset}_{args.attack}_seed{args.seed}.json",
    )
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint":  args.checkpoint,
            "dataset":     args.dataset,
            "attack":      args.attack,
            "n_samples":   args.n_samples,
            "ig_steps":    args.ig_steps,
            "seed":        args.seed,
            "aua":         aua,
            "n_total":     n_total,
            "n_flipped":   n_flipped,
            "s_adv":       s_adv,
            "n_stable":    n_stable,
        }, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── 4. Vẽ heatmap (tuần 5–6: --visualize 10) ─────────────────────────
    if args.visualize > 0 and heatmap_data:
        from scipy.stats import spearmanr
        viz_dir = os.path.join(args.output_dir, f"ig_heatmaps_{ckpt_tag}_{args.attack}")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"\nSaving {len(heatmap_data)} heatmaps → {viz_dir}/")

        for idx, (tokens_c, attr_c, tokens_a, attr_a, rec) in enumerate(heatmap_data):
            pad_tokens = ("[PAD]", "<pad>")
            non_pad = next(
                (i for i, t in enumerate(tokens_c) if t in pad_tokens),
                len(tokens_c)
            )
            rho = float("nan")
            if not rec["pred_flipped"] and non_pad > 1:
                rho, _ = spearmanr(attr_c[:non_pad], attr_a[:non_pad])
            save_heatmap(
                tokens_c, attr_c, tokens_a, attr_a,
                rec["label"], rec["pred_flipped"], rho,
                idx + 1, viz_dir,
            )

    print(f"\n{'─'*40}")
    print(f"  AUA   : {aua:.4f}")
    print(f"  S_adv : {s_adv:.4f}")
    print(f"{'─'*40}")


if __name__ == "__main__":
    main()