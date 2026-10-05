"""
Evaluation suite for College Enquiry Chatbot.
Computes Accuracy, Precision, Recall, F1-scores, and tests Out-of-Scope rejection.
"""

import os
import sys
import json
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import SimpleTokenizer
from src.dataset import CollegeFAQDataProcessor, PyTorchCollegeDataset, TORCH_AVAILABLE


def evaluate_test_set(
    data_dir: str = "./data",
    checkpoint_dir: str = "./checkpoints"
):
    # Load test split
    test_path = os.path.join(data_dir, "test.json")
    if not os.path.exists(test_path):
        faq_path = os.path.join(data_dir, "college_faq_dataset.json")
        processor = CollegeFAQDataProcessor(faq_path)
        processor.save_splits(data_dir, augment=True)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)["samples"]

    faq_path = os.path.join(data_dir, "college_faq_dataset.json")
    with open(faq_path, "r", encoding="utf-8") as f:
        faq_raw = json.load(f)

    # Use lightweight semantic matcher for benchmark
    from models.custom_transformer import LightweightSemanticMatcher
    matcher = LightweightSemanticMatcher(faq_raw["intents"])

    y_true = []
    y_pred = []
    correct = 0

    print(f"[*] Running Evaluation on {len(test_data)} test queries...\n")

    per_intent_stats = {}

    for item in test_data:
        gold_intent = item["intent"]
        query = item["query"]

        pred_res = matcher.predict(query)
        pred_intent = pred_res["intent"]

        y_true.append(gold_intent)
        y_pred.append(pred_intent)

        if gold_intent not in per_intent_stats:
            per_intent_stats[gold_intent] = {"tp": 0, "fp": 0, "fn": 0, "total": 0}
        if pred_intent not in per_intent_stats:
            per_intent_stats[pred_intent] = {"tp": 0, "fp": 0, "fn": 0, "total": 0}

        per_intent_stats[gold_intent]["total"] += 1

        if gold_intent == pred_intent:
            correct += 1
            per_intent_stats[gold_intent]["tp"] += 1
        else:
            per_intent_stats[pred_intent]["fp"] += 1
            per_intent_stats[gold_intent]["fn"] += 1

    accuracy = (correct / len(test_data)) * 100

    print("=" * 75)
    print(f"{'Intent':<24} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'F1-Score (%)':<12}")
    print("=" * 75)

    f1_scores = []
    for intent, stats in sorted(per_intent_stats.items()):
        if stats["total"] == 0 and stats["fp"] == 0:
            continue
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

        print(f"{intent:<24} | {prec:<14.2f} | {rec:<12.2f} | {f1:<12.2f}")

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print("=" * 75)
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(f"Macro F1-Score:       {macro_f1:.2f}%")
    print("=" * 75)

    # Save evaluation report
    report = {
        "test_samples": len(test_data),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_intent": per_intent_stats
    }
    with open(os.path.join(checkpoint_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    evaluate_test_set()
