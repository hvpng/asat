# training/asat.py
# ASAT: Attribution-Stable Adversarial Training
#
# Objective:
#   L_total = L_cls + λ1·L_adv + λ2·L_align
#
#   L_cls   : cross-entropy trên clean input
#   L_adv   : FreeLB — cross-entropy trên worst-case perturbed input (K steps)
#   L_align : MSE( ā_clean, ā_adv ) — ép attribution map nhất quán
#             ā = L2-normalize GradxInput theo chiều embedding (dim=-1)
#
# Kỹ thuật quan trọng:
#   - GradxInput dùng create_graph=True → double backprop qua L_align
#   - Attribution tính theo ground-truth label y (target-consistent)
#   - L2-norm theo dim=-1 (per token, per hidden dim) trước khi MSE
#   - Gradient accumulation qua K FreeLB steps (giống freelb.py)
#
# Usage:
#   python training/asat.py --dataset sst2 --seed 42 --no_wandb
# Resume:
#   python training/asat.py --dataset sst2 --seed 42 --resume --no_wandb

import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from data.load_data import load_and_tokenize
from attribution.grad_input import compute_grad_input

RESUME_STATE_FILE = "training_state.pt"


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ASAT adversarial training")
    p.add_argument("--dataset",       type=str,   default="sst2",
                   choices=["sst2", "imdb", "yelp"])
    p.add_argument("--model_name",    type=str,   default="bert-base-uncased")
    p.add_argument("--max_length",    type=int,   default=128)
    p.add_argument("--num_epochs",    type=int,   default=3)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed",          type=int,   default=42)
    # FreeLB hyperparams (giữ nguyên từ freelb.py)
    p.add_argument("--freelb_k",      type=int,   default=3)
    p.add_argument("--freelb_eps",    type=float, default=1e-6)
    p.add_argument("--freelb_alpha",  type=float, default=3e-1)
    # ASAT hyperparams
    p.add_argument("--lambda1",       type=float, default=1.0,
                   help="Weight của L_adv")
    p.add_argument("--lambda2",       type=float, default=0.5,
                   help="Weight của L_align. Ablation: thử 0.1, 0.5, 1.0, 2.0")
    p.add_argument("--eps_norm",      type=float, default=1e-8,
                   help="Epsilon trong L2-norm để tránh chia 0")
    p.add_argument("--train_subset",  type=int,   default=None)
    p.add_argument("--output_dir",    type=str,   default="checkpoints/asat")
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--resume",        action="store_true")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_embedding_layer(model):
    if hasattr(model, "bert"):
        return model.bert.embeddings.word_embeddings
    elif hasattr(model, "roberta"):
        return model.roberta.embeddings.word_embeddings
    else:
        raise ValueError(f"Unsupported architecture: {type(model)}")


def l2_normalize_attribution(gi, eps):
    """
    L2-normalize GradxInput theo chiều embedding (dim=-1).
    gi : Tensor [batch, seq_len, hidden_dim]
    Returns ā : Tensor [batch, seq_len, hidden_dim], unit-norm per token
    """
    norm = gi.norm(dim=-1, keepdim=True).clamp(min=eps)
    return gi / norm


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            preds = model(input_ids=input_ids,
                          attention_mask=attention_mask).logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


def save_training_state(output_dir, optimizer, scheduler, epoch, best_val_acc):
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "last_completed_epoch": epoch,
        "best_val_acc":         best_val_acc,
    }, os.path.join(output_dir, RESUME_STATE_FILE))
    print(f"  [resume] State saved → {output_dir}/{RESUME_STATE_FILE}", flush=True)


def load_training_state(output_dir, optimizer, scheduler):
    path = os.path.join(output_dir, RESUME_STATE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No training state at {path}. Run without --resume to start fresh."
        )
    state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    start_epoch  = state["last_completed_epoch"] + 1
    best_val_acc = state["best_val_acc"]
    print(
        f"  [resume] Loaded from {path}\n"
        f"  [resume] Next epoch = {start_epoch + 1} | best_val_acc = {best_val_acc:.4f}",
        flush=True,
    )
    return start_epoch, best_val_acc


# ── ASAT training step ────────────────────────────────────────────────────

def asat_step(model, batch, device, args):
    """
    Một ASAT training step.

    Luồng xử lý:
      1. Lấy clean embeddings Z_clean
      2. Tính GI_clean = GradxInput(Z_clean, y)  [create_graph=True]
      3. FreeLB K steps: tìm worst-case delta δ*
         - Mỗi step accumulate gradient vào θ (giống freelb.py)
         - Step cuối: lấy Z_adv = Z_clean + δ*
      4. Tính GI_adv = GradxInput(Z_adv, y)      [create_graph=True]
      5. L_align = MSE( ā_clean, ā_adv )          [ā = L2-norm GI]
      6. loss_align = λ2 * L_align, backward      [double backprop]

    Tổng loss được accumulate đúng như FreeLB:
      L_total = (1/K) * Σ L_adv_k  +  λ2 * L_align
    L_cls được tính ngầm trong L_adv ở step đầu (delta=0).

    Returns:
        total_loss : float (để log)
        align_loss : float (để monitor xem L_align có hội tụ không)
    """
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["label"].to(device)

    embedding_layer = get_embedding_layer(model)

    # Z_clean: detach khỏi embedding layer, sau đó requires_grad để tính GI
    Z_clean = embedding_layer(input_ids).detach().requires_grad_(True)

    # ── Bước 2: GI_clean ─────────────────────────────────────────────────
    # compute_grad_input cần model ở train mode và create_graph=True bên trong
    GI_clean = compute_grad_input(model, Z_clean, attention_mask, labels)
    # [batch, seq_len, hidden_dim]

    # ── Bước 3: FreeLB K steps ───────────────────────────────────────────
    delta = torch.zeros_like(Z_clean, requires_grad=True)
    total_loss = 0.0

    for step_k in range(args.freelb_k):
        Z_adv = Z_clean.detach() + delta  # tách Z_clean để delta là variable duy nhất

        outputs = model(
            inputs_embeds  = Z_adv,
            attention_mask = attention_mask,
            labels         = labels,
        )
        loss_adv = outputs.loss / args.freelb_k

        # Accumulate vào θ (giữ graph cho step tiếp theo nếu cần)
        loss_adv.backward(retain_graph=True)
        total_loss += loss_adv.item()

        if step_k < args.freelb_k - 1:
            # Gradient ascent trên delta
            delta_grad = delta.grad.detach()
            delta = delta + args.freelb_alpha * delta_grad.sign()
            delta_norm = delta.data.norm(p="fro")
            if delta_norm > args.freelb_eps:
                delta = delta * (args.freelb_eps / delta_norm)
            delta = delta.detach().requires_grad_(True)

    # Z_adv tại worst-case delta (step cuối)
    Z_adv_final = Z_clean.detach() + delta.detach()
    Z_adv_final = Z_adv_final.detach().requires_grad_(True)

    # ── Bước 4: GI_adv ───────────────────────────────────────────────────
    GI_adv = compute_grad_input(model, Z_adv_final, attention_mask, labels)

    # ── Bước 5: L_align = MSE( ā_clean, ā_adv ) ─────────────────────────
    a_clean = l2_normalize_attribution(GI_clean, args.eps_norm)
    a_adv   = l2_normalize_attribution(GI_adv,   args.eps_norm)

    L_align = F.mse_loss(a_clean, a_adv)

    # ── Bước 6: Double backprop qua L_align ──────────────────────────────
    # create_graph=True trong compute_grad_input đã giữ đồ thị bậc 2
    loss_align = args.lambda2 * L_align
    loss_align.backward()

    return total_loss, L_align.item()


# ── Training loop ─────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}", flush=True)
    print(f"λ1={args.lambda1}  λ2={args.lambda2}  K={args.freelb_k}", flush=True)

    if not args.no_wandb:
        import wandb
        wandb.init(
            project="asat-research",
            name=f"asat_{args.dataset}_lam2{args.lambda2}_seed{args.seed}",
            config=vars(args),
            resume="allow",
        )

    splits, tokenizer = load_and_tokenize(
        args.dataset,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        train_subset=args.train_subset,
    )
    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(splits["val"],   batch_size=args.batch_size * 2)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume and os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"  [resume] Loading model from {args.output_dir}", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2
        ).to(device)

    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    start_epoch  = 0
    best_val_acc = 0.0
    if args.resume:
        start_epoch, best_val_acc = load_training_state(args.output_dir, optimizer, scheduler)
        if start_epoch >= args.num_epochs:
            print(f"Already completed {args.num_epochs} epochs.", flush=True)
            return best_val_acc

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss_epoch  = 0.0
        total_align_epoch = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            step_loss, step_align = asat_step(model, batch, device, args)
            total_loss_epoch  += step_loss
            total_align_epoch += step_align

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                avg_loss  = total_loss_epoch  / (step + 1)
                avg_align = total_align_epoch / (step + 1)
                print(
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | L_align: {avg_align:.4f}",
                    flush=True,
                )
                if not args.no_wandb:
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/L_align": avg_align})

        val_acc = evaluate(model, val_loader, device)
        avg_loss  = total_loss_epoch  / len(train_loader)
        avg_align = total_align_epoch / len(train_loader)

        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch+1} | λ2={args.lambda2}", flush=True)
        print(f"  Train loss : {avg_loss:.4f}", flush=True)
        print(f"  L_align    : {avg_align:.4f}", flush=True)
        print(f"  Val acc    : {val_acc*100:.2f}%", flush=True)
        print(f"{'='*60}\n", flush=True)

        if not args.no_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/avg_loss":  avg_loss,
                "train/avg_align": avg_align,
                "eval/clean_acc":  val_acc,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  Saved best model (val_acc={val_acc:.4f})", flush=True)

        save_training_state(args.output_dir, optimizer, scheduler, epoch, best_val_acc)

    os.makedirs("results", exist_ok=True)
    path = f"results/asat_{args.dataset}_lam2{args.lambda2}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump({
            "method":      "asat",
            "dataset":     args.dataset,
            "model":       args.model_name,
            "seed":        args.seed,
            "lambda1":     args.lambda1,
            "lambda2":     args.lambda2,
            "freelb_k":    args.freelb_k,
            "best_val_acc": best_val_acc,
        }, f, indent=2)

    print(f"Best val acc : {best_val_acc:.4f}", flush=True)
    print(f"Results      : {path}", flush=True)

    if not args.no_wandb:
        import wandb
        wandb.finish()

    return best_val_acc


if __name__ == "__main__":
    args = parse_args()
    train(args)