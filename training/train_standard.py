# training/train_standard.py
# Standard BERT fine-tuning baseline (no adversarial training).
# Usage: python training/train_standard.py --dataset sst2 --seed 42

import argparse
import os
import sys
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

# Add project root to sys.path so "data.load_data" is importable
# Works whether you run as: python training/train_standard.py
# or from project root: python -m training.train_standard
_this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
_root = os.path.dirname(_this_dir) if os.path.basename(_this_dir) == "training" else _this_dir
if _root not in sys.path:
    sys.path.insert(0, _root)

from data.load_data import load_and_tokenize


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--train_subset",  type=int,   default=None)
    parser.add_argument("--output_dir",    type=str,   default="checkpoints/standard")
    parser.add_argument("--no_wandb",      action="store_true")
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
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if not args.no_wandb:
        import wandb
        wandb.init(
            project="asat-research",
            name=f"standard_{args.dataset}_seed{args.seed}",
            config=vars(args),
        )

    # Load data
    splits, tokenizer = load_and_tokenize(
        args.dataset,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        train_subset=args.train_subset,
    )
    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(splits["val"],   batch_size=args.batch_size * 2)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    # Optimizer + scheduler
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % 50 == 0:
                avg = total_loss / (step + 1)
                print(
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss: {avg:.4f}",
                    flush=True,
                )

        # Eval after each epoch
        val_acc, _, _ = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)

        print("=" * 60, flush=True)
        print(f"Epoch {epoch+1} done", flush=True)
        print(f"  Train loss : {avg_loss:.4f}", flush=True)
        print(f"  Val acc    : {val_acc*100:.2f}%", flush=True)
        print("=" * 60, flush=True)

        if not args.no_wandb:
            import wandb
            wandb.log({"epoch": epoch+1, "eval/clean_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best model  val_acc={val_acc:.4f}", flush=True)

    # Save results JSON
    os.makedirs("results", exist_ok=True)
    path = f"results/standard_{args.dataset}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump({
            "method": "standard",
            "dataset": args.dataset,
            "model": args.model_name,
            "seed": args.seed,
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