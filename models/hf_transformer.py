"""
Hugging Face Pretrained Transformer Fine-Tuning & Inference Pipeline.
Supports DistilBERT, BERT, RoBERTa, and MiniLM for state-of-the-art College Enquiry Intent Classification.
"""

import os
from typing import Dict, List, Optional, Tuple

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        pipeline
    )
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HuggingFaceIntentClassifier:
    """
    Wraps Hugging Face Transformers for fine-tuning and inference on College FAQ intents.
    Default model: 'distilbert-base-uncased' (lightweight, fast, high accuracy).
    """

    def __init__(self, model_name_or_path: str = "distilbert-base-uncased", num_labels: int = 23):
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        if HF_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    num_labels=num_labels
                )
            except Exception as e:
                print(f"[Notice] Hugging Face model loading deferred: {e}")

    def is_available(self) -> bool:
        return HF_AVAILABLE and self.model is not None

    def predict(self, text: str, id2label: Optional[Dict[int, str]] = None) -> Dict:
        """Runs Transformer inference to predict intent and confidence."""
        if not self.is_available():
            raise RuntimeError("Hugging Face Transformers or model weights are not loaded.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_id].item()

        intent_name = id2label[pred_id] if id2label else f"intent_{pred_id}"
        return {
            "intent": intent_name,
            "intent_id": pred_id,
            "confidence": round(confidence, 4)
        }

    def train_model(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        output_dir: str = "./checkpoints/hf_distilbert",
        epochs: int = 5,
        batch_size: int = 16
    ):
        """Fine-tunes the Pretrained Transformer using Hugging Face Trainer."""
        if not HF_AVAILABLE:
            raise RuntimeError("Transformers library is required for fine-tuning.")

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=64)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=64)

        class IntentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_ds = IntentDataset(train_encodings, train_labels)
        val_ds = IntentDataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Hugging Face fine-tuned model saved successfully to: {output_dir}")
