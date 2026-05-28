# training/freelb.py
# FreeLB: Enhanced Adversarial Training for NLP
# Zhu et al., ICLR 2020 - https://arxiv.org/abs/1902.03932
#
# Usage: python training/freelb.py --dataset sst2 --seed 42
# Resume: python training/freelb.py --dataset sst2 --seed 42 --resume

import argparse
import os
import json
import torch
import numpy as np

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import load_and_tokenize

RESUME_STATE_FILE = "training_state.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="FreeLB adversarial training")
    parser.add_argument("--dataset",       type=str,   default="sst2",
                        choices=["sst2", "imdb", "yelp"])
    parser.add_argument("--model_name",    type=str,   default="bert-base-uncased")
    parser.add_argument("--max_length",    type=int,   default=128)
    parser.add_argument("--num_epochs",    type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",  type=float, default=0.1)
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--freelb_k",     type=int,   default=3)
    parser.add_argument("--freelb_eps",   type=float, default=1e-6)
    parser.add_argument("--freelb_alpha", type=float, default=3e-1)
    parser.add_argument("--lambda1",      type=float, default=1.0)
    parser.add_argument("--train_subset", type=int,   default=None)
    parser.add_argument("--output_dir",   type=str,   default="checkpoints/freelb")
    parser.add_argument("--no_wandb",     action="store_true")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from last checkpoint in --output_dir")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds), all_preds, all_labels


def get_word_embeddings(model):
    if hasattr(model, 'bert'):
        return model.bert.embeddings.word_embeddings
    elif hasattr(model, 'roberta'):
        return model.roberta.embeddings.word_embeddings
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")


def freelb_step(model, batch, device, args):
    """
    One FreeLB training step.
    Gradients are accumulated into theta across all K ascent steps.
    """
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["label"].to(device)

    embedding_layer = get_word_embeddings(model)
    embeddings = embedding_layer(input_ids).detach()

    delta = torch.zeros_like(embeddings, requires_grad=True)
    total_loss = 0.0

    for step_k in range(args.freelb_k):
        outputs = model(
            inputs_embeds  = embeddings + delta,
            attention_mask = attention_mask,
            labels         = labels,
        )
        loss = outputs.loss / args.freelb_k
        loss.backward(retain_graph=(step_k < args.freelb_k - 1))
        total_loss += loss.item()

        if step_k < args.freelb_k - 1:
            delta_grad = delta.grad.detach()
            delta = delta + args.freelb_alpha * delta_grad.sign()
            delta_norm = delta.data.norm(p="fro")
            if delta_norm > args.freelb_eps:
                delta = delta * (args.freelb_eps / delta_norm)
            delta = delta.detach().requires_grad_(True)

    return total_loss


def save_training_state(output_dir, optimizer, scheduler, epoch, best_val_acc):
    """Save optimizer/scheduler state after each epoch for exact resume."""
    state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "last_completed_epoch": epoch,
        "best_val_acc": best_val_acc,
    }
    path = os.path.join(output_dir, RESUME_STATE_FILE)
    torch.save(state, path)
    print(f"  [resume] State saved → {path}", flush=True)


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


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if not args.no_wandb:
        import wandb
        wandb.init(
            project="asat-research",
            name=f"freelb_{args.dataset}_seed{args.seed}",
            config=vars(args),
            resume="allow",
        )

    splits, tokenizer = load_and_tokenize(
        args.dataset,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        train_subset=args.train_subset,
    )
    train_loader = DataLoader(
        splits["train"], batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        splits["val"], batch_size=args.batch_size * 2
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume and os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"  [resume] Loading model from {args.output_dir}", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2
        ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_epoch  = 0
    best_val_acc = 0.0
    if args.resume:
        start_epoch, best_val_acc = load_training_state(args.output_dir, optimizer, scheduler)
        if start_epoch >= args.num_epochs:
            print(
                f"Already completed {args.num_epochs} epoch(s). "
                "Increase --num_epochs to continue.",
                flush=True,
            )
            return best_val_acc

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            step_loss = freelb_step(model, batch, device, args)
            total_loss += step_loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                avg = total_loss / (step + 1)
                print(
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss: {avg:.4f}",
                    flush=True,
                )
                if not args.no_wandb:
                    import wandb
                    wandb.log({"train/loss": avg})

        val_acc, _, _ = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)

        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch+1} done | FreeLB K={args.freelb_k} eps={args.freelb_eps}", flush=True)
        print(f"  Train loss : {avg_loss:.4f}", flush=True)
        print(f"  Val acc    : {val_acc:.4f} ({val_acc*100:.2f}%)", flush=True)
        print(f"{'='*60}\n", flush=True)

        if not args.no_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/avg_loss": avg_loss,
                "eval/clean_acc": val_acc,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  Saved best model (val_acc={val_acc:.4f})", flush=True)

        save_training_state(args.output_dir, optimizer, scheduler, epoch, best_val_acc)

    results = {
        "method": "freelb",
        "dataset": args.dataset,
        "model": args.model_name,
        "seed": args.seed,
        "freelb_k": args.freelb_k,
        "freelb_eps": args.freelb_eps,
        "best_val_acc": best_val_acc,
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/freelb_{args.dataset}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest val accuracy : {best_val_acc:.4f}", flush=True)
    print(f"Results saved to  : {path}", flush=True)

    if not args.no_wandb:
        import wandb
        wandb.finish()

    return best_val_acc


if __name__ == "__main__":
    args = parse_args()
    train(args)