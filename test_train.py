import sys, os
sys.path.insert(0, ".")
print("step 1", flush=True)

import argparse, json, torch, numpy as np
print("step 2", flush=True)

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
print("step 3", flush=True)

from data.load_data import load_and_tokenize
print("step 4", flush=True)

args_dataset = "sst2"
args_train_subset = 500
args_model_name = "bert-base-uncased"
splits, tokenizer = load_and_tokenize(args_dataset, train_subset=args_train_subset)
print("step 5 data loaded", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("step 6 device:", device, flush=True)

model = AutoModelForSequenceClassification.from_pretrained(args_model_name, num_labels=2).to(device)
print("step 7 model loaded", flush=True)
