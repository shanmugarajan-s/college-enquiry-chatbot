"""
Custom Vocabulary & Tokenizer for Transformer-based College Enquiry Chatbot.
Provides subword/word tokenization, vocabulary mapping, padding, and attention mask creation.
"""

import re
from typing import List, Dict, Tuple, Optional


class SimpleTokenizer:
    """
    A lightweight, robust tokenizer with special tokens support:
    [PAD] -> 0, [UNK] -> 1, [CLS] -> 2, [SEP] -> 3, [MASK] -> 4
    """

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_length: int = 32):
        self.max_length = max_length
        if vocab is not None:
            self.vocab = vocab
            self.inv_vocab = {idx: token for token, idx in vocab.items()}
        else:
            self.vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
            self.inv_vocab = {idx: token for idx, token in enumerate(self.SPECIAL_TOKENS)}

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]

    @property
    def cls_token_id(self) -> int:
        return self.vocab[self.CLS_TOKEN]

    @property
    def sep_token_id(self) -> int:
        return self.vocab[self.SEP_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalizes casing, contractions, and punctuation spacing."""
        text = text.lower().strip()
        # Separate punctuation with spaces for clean tokenization
        text = re.sub(r"([?.!,;:'\"()\[\]{}&/%$#@+-])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        return cleaned.split()

    def build_vocab(self, texts: List[str], min_freq: int = 1, max_vocab_size: int = 5000):
        """Builds vocabulary from a corpus of texts with frequency thresholding."""
        word_counts: Dict[str, int] = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        # Sort tokens by frequency
        sorted_tokens = sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))

        # Preserve special tokens
        self.vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        next_idx = len(self.SPECIAL_TOKENS)

        for token, count in sorted_tokens:
            if count >= min_freq and next_idx < max_vocab_size:
                if token not in self.vocab:
                    self.vocab[token] = next_idx
                    next_idx += 1

        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> Dict[str, List[int]]:
        """
        Encodes a string into input_ids and attention_mask.
        Returns:
            {
                "input_ids": List[int],
                "attention_mask": List[int],
                "tokens": List[str]
            }
        """
        raw_tokens = self.tokenize(text)
        tokens = []

        if add_special_tokens:
            tokens.append(self.CLS_TOKEN)
            # Truncate to fit [CLS] and [SEP] within max_length
            available_len = self.max_length - 2
            tokens.extend(raw_tokens[:available_len])
            tokens.append(self.SEP_TOKEN)
        else:
            tokens = raw_tokens[:self.max_length]

        input_ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tokens": tokens
        }

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in ids:
            tok = self.inv_vocab.get(token_id, self.UNK_TOKEN)
            if skip_special_tokens and tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def to_dict(self) -> Dict:
        return {
            "vocab": self.vocab,
            "max_length": self.max_length
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SimpleTokenizer":
        tokenizer = cls(vocab=data.get("vocab"), max_length=data.get("max_length", 32))
        return tokenizer
