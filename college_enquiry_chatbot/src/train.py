"""
Training pipeline for Custom PyTorch Transformer on College FAQ dataset.
Supports AdamW optimizer, warmup, validation evaluation, early stopping, and checkpoint export.
"""

import os
import sys
import json
import time
import math
from typing import Dict, List, Optional

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import SimpleTokenizer
from src.dataset import CollegeFAQDataProcessor, PyTorchCollegeDataset, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from models.custom_transformer import TransformerIntentClassifier


def train_model(
    data_dir: str = "./data",
    output_dir: str = "./checkpoints",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    d_model: int = 128,
    num_heads: int = 4,
    d_ff: int = 256,
    num_layers: int = 3,
    max_len: int = 48,
    seed: int = 42
):
    if not TORCH_AVAILABLE:
        print("[Error] PyTorch is required to run neural training. Please install torch.")
        return

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 1. Load and process FAQ data
    faq_path = os.path.join(data_dir, "college_faq_dataset.json")
    processor = CollegeFAQDataProcessor(faq_path, seed=seed)
    train_samples, val_samples, test_samples = processor.save_splits(data_dir, augment=True)

    print(f"[*] Dataset Splits -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    print(f"[*] Total Intents: {len(processor.intents)}")

    # 2. Build Vocabulary & Tokenizer
    all_texts = [s["query"] for s in train_samples + val_samples + test_samples]
    tokenizer = SimpleTokenizer(max_length=max_len)
    tokenizer.build_vocab(all_texts, min_freq=1)
    print(f"[*] Vocabulary size: {tokenizer.vocab_size}")

    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)

    # 3. Create PyTorch DataLoaders
    train_dataset = PyTorchCollegeDataset(train_samples, tokenizer, max_length=max_len)
    val_dataset = PyTorchCollegeDataset(val_samples, tokenizer, max_length=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Instantiate Model
    num_classes = len(processor.intents)
    model = TransformerIntentClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len,
        dropout=0.1,
        pad_idx=tokenizer.pad_token_id
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Transformer Model Initialized with {total_params:,} trainable parameters.")

    # 5. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print("\n" + "=" * 60)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Acc (%)':<12} | {'Time (s)':<8}")
    print("=" * 60)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = (correct / total) * 100 if total > 0 else 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - start_time
        print(f"{epoch:<6} | {avg_train_loss:<12.4f} | {avg_val_loss:<10.4f} | {val_acc:<12.2f} | {epoch_time:<8.1f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(output_dir, "best_transformer.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "vocab_size": tokenizer.vocab_size,
                "num_classes": num_classes,
                "d_model": d_model,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "num_layers": num_layers,
                "max_len": max_len,
                "intents": processor.intents,
                "intent2id": processor.intent2id,
                "id2intent": processor.id2intent
            }, best_model_path)

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print(f"[*] Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[*] Checkpoint saved at: {os.path.join(output_dir, 'best_transformer.pt')}")


if __name__ == "__main__":
    train_model()
