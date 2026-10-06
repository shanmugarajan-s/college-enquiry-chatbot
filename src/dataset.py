"""
Dataset loader and Bitext / CLINC-style synthetic query expansion for College Enquiry Chatbot.
Generates train, validation, and test splits with balanced samples.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional

# Optional PyTorch import handled gracefully
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object
    DataLoader = None


# Templates for Bitext/CLINC-style intent expansion
QUERY_PREFIXES = [
    "",
    "please tell me ",
    "can you tell me ",
    "could you please explain ",
    "i want to know about ",
    "i would like to enquire about ",
    "can you provide information on ",
    "do you have details regarding ",
    "i need help with ",
    "what can you tell me about ",
    "excuse me, ",
    "hey bot, "
]

QUERY_SUFFIXES = [
    "",
    " please",
    " for my admission",
    " in this college",
    " at apex institute",
    " for first year",
    " right now"
]

SYNONYM_REPLACEMENTS = {
    "courses": ["programs", "degrees", "academic courses"],
    "fees": ["charges", "expenses", "tuition cost", "fee structure"],
    "exams": ["examinations", "tests", "assessments"],
    "departments": ["branches", "schools", "academic departments"],
    "timings": ["schedule", "working hours", "timing"],
    "hostel": ["accommodation", "dormitory", "hostel rooms"],
    "library": ["knowledge center", "central library", "reading hall"],
    "wifi": ["internet", "campus network", "wireless connectivity"]
}


def expand_queries(pattern: str, num_variations: int = 4) -> List[str]:
    """Generates synthetic query paraphrases inspired by Bitext/CLINC data augmentation."""
    variations = {pattern}
    base = pattern.strip()

    # Punctuation stripped version
    clean = base.rstrip("?.!")

    for _ in range(num_variations * 2):
        prefix = random.choice(QUERY_PREFIXES)
        suffix = random.choice(QUERY_SUFFIXES)

        text = clean
        # Occasional synonym replacement
        for word, syns in SYNONYM_REPLACEMENTS.items():
            if word in text.lower() and random.random() < 0.4:
                replacement = random.choice(syns)
                text = text.replace(word, replacement)

        aug_query = f"{prefix}{text}{suffix}".strip()
        if aug_query:
            # Capitalize first letter
            aug_query = aug_query[0].upper() + aug_query[1:]
            if not aug_query.endswith("?"):
                aug_query += random.choice(["?", ""])
            variations.add(aug_query)

        if len(variations) >= num_variations + 1:
            break

    return list(variations)


class CollegeFAQDataProcessor:
    """Processes the raw FAQ dataset, generates intents, expands queries, and splits data."""

    def __init__(self, raw_data_path: str, seed: int = 42):
        self.raw_data_path = raw_data_path
        self.seed = seed
        random.seed(seed)
        with open(raw_data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.intents = [item["intent"] for item in self.raw_data["intents"]]
        self.intent2id = {intent: idx for idx, intent in enumerate(self.intents)}
        self.id2intent = {idx: intent for idx, intent in enumerate(self.intents)}
        self.categories = self.raw_data.get("categories", [])
        self.intent_to_response = {
            item["intent"]: item["response"] for item in self.raw_data["intents"]
        }
        self.intent_to_suggestions = {
            item["intent"]: item.get("suggestions", []) for item in self.raw_data["intents"]
        }
        self.intent_to_category = {
            item["intent"]: item.get("category", "general") for item in self.raw_data["intents"]
        }

    def generate_all_samples(self, augment: bool = True) -> List[Dict]:
        """Generates all query-intent-category samples with optional augmentation."""
        samples = []
        for item in self.raw_data["intents"]:
            intent = item["intent"]
            category = item.get("category", "general")
            patterns = item["patterns"]

            seen_queries = set()
            for p in patterns:
                p_clean = p.strip()
                if p_clean not in seen_queries:
                    seen_queries.add(p_clean)
                    samples.append({
                        "query": p_clean,
                        "intent": intent,
                        "intent_id": self.intent2id[intent],
                        "category": category
                    })

                if augment:
                    expansions = expand_queries(p_clean, num_variations=3)
                    for exp in expansions:
                        if exp not in seen_queries:
                            seen_queries.add(exp)
                            samples.append({
                                "query": exp,
                                "intent": intent,
                                "intent_id": self.intent2id[intent],
                                "category": category
                            })
        return samples

    def split_data(
        self,
        samples: List[Dict],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Stratified train/val/test split across all intent categories."""
        by_intent: Dict[str, List[Dict]] = {}
        for s in samples:
            by_intent.setdefault(s["intent"], []).append(s)

        train_set, val_set, test_set = [], [], []

        for intent, items in by_intent.items():
            random.shuffle(items)
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_items = items[:n_train]
            val_items = items[n_train:n_train + n_val]
            test_items = items[n_train + n_val:]

            # Guarantee at least 1 sample in train and test if possible
            if not val_items and len(train_items) > 2:
                val_items.append(train_items.pop())
            if not test_items and len(train_items) > 2:
                test_items.append(train_items.pop())

            train_set.extend(train_items)
            val_set.extend(val_items)
            test_set.extend(test_items)

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)
        return train_set, val_set, test_set

    def save_splits(self, output_dir: str, augment: bool = True):
        """Generates and writes train.json, val.json, and test.json files."""
        os.makedirs(output_dir, exist_ok=True)
        samples = self.generate_all_samples(augment=augment)
        train_s, val_s, test_s = self.split_data(samples)

        for filename, data in [
            ("train.json", train_s),
            ("val.json", val_s),
            ("test.json", test_s)
        ]:
            out_file = os.path.join(output_dir, filename)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({
                    "count": len(data),
                    "samples": data
                }, f, indent=2)

        meta_file = os.path.join(output_dir, "metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "num_intents": len(self.intents),
                "intents": self.intents,
                "intent2id": self.intent2id,
                "id2intent": self.id2intent,
                "categories": self.categories,
                "total_samples": len(samples),
                "train_samples": len(train_s),
                "val_samples": len(val_s),
                "test_samples": len(test_s)
            }, f, indent=2)

        return train_s, val_s, test_s


if TORCH_AVAILABLE:
    class PyTorchCollegeDataset(Dataset):
        """PyTorch Dataset wrapper for college intent queries."""

        def __init__(self, samples: List[Dict], tokenizer, max_length: int = 32):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            item = self.samples[idx]
            encoded = self.tokenizer.encode(item["query"], add_special_tokens=True)
            return {
                "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.float),
                "label": torch.tensor(item["intent_id"], dtype=torch.long),
                "query": item["query"],
                "intent": item["intent"]
            }
