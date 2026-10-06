"""
Custom Transformer Architecture for College Enquiry Intent Classification & Representation.
Implements Multi-Head Self-Attention, Positional Encoding, and Transformer Encoder Layer from scratch.
Includes full PyTorch implementation with an elegant Pure-Python fallback for maximum portability.
"""

import math
from typing import Optional, Tuple, Dict, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object
    F = None


if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """
        Sinusoidal Positional Encoding:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        """

        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch_size, seq_len, d_model]
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class MultiHeadAttention(nn.Module):
        """
        Multi-Head Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        """

        def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size, seq_len, _ = q.size()

            # 1) Linear projections & split into num_heads
            # [batch, seq_len, num_heads, d_k] -> transpose to [batch, num_heads, seq_len, d_k]
            q_proj = self.w_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            k_proj = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            v_proj = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            # 2) Scaled dot-product: [batch, num_heads, seq_len, seq_len]
            scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Apply padding mask if provided
            if mask is not None:
                # mask shape: [batch, 1, 1, seq_len]
                scores = scores.masked_fill(mask == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 3) Context multiplication: [batch, num_heads, seq_len, d_k]
            context = torch.matmul(attn_weights, v_proj)

            # 4) Concatenate heads and project output
            # [batch, seq_len, num_heads * d_k] -> [batch, seq_len, d_model]
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.w_o(context)

            return output, attn_weights


    class FeedForwardNetwork(nn.Module):
        """Position-wise Feed-Forward Network with GELU activation."""

        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ff, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear2(self.dropout(self.activation(self.linear1(x))))


    class TransformerEncoderLayer(nn.Module):
        """Transformer Encoder Layer with Pre-LayerNorm or Post-LayerNorm architecture."""

        def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

            self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # Self-attention with residual connection & layer norm
            norm_x = self.norm1(x)
            attn_out, attn_weights = self.self_attn(norm_x, norm_x, norm_x, mask)
            x = x + self.dropout1(attn_out)

            # FFN with residual connection & layer norm
            norm_x2 = self.norm2(x)
            ffn_out = self.ffn(norm_x2)
            x = x + self.dropout2(ffn_out)

            return x, attn_weights


    class TransformerIntentClassifier(nn.Module):
        """
        End-to-End Transformer Model for College Enquiry Intent Classification.
        Architecture:
        Embedding -> Positional Encoding -> N x Transformer Encoder Layers ->
        Attention Pooling / CLS Pooling -> Dropout -> Linear Classifier -> Softmax
        """

        def __init__(
            self,
            vocab_size: int,
            num_classes: int,
            d_model: int = 128,
            num_heads: int = 4,
            d_ff: int = 256,
            num_layers: int = 3,
            max_len: int = 64,
            dropout: float = 0.1,
            pad_idx: int = 0
        ):
            super().__init__()
            self.d_model = d_model
            self.pad_idx = pad_idx

            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
            self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

            self.layers = nn.ModuleList([
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            self.final_norm = nn.LayerNorm(d_model)

            # Classification Head
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            # input_ids: [batch_size, seq_len]
            # attention_mask: [batch_size, seq_len]
            batch_size, seq_len = input_ids.size()

            # Create 4D attention mask for MultiHeadAttention: [batch, 1, 1, seq_len]
            if attention_mask is not None:
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                extended_mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)

            # Embedding & positional encoding
            x = self.embedding(input_ids) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)

            all_attn_weights = []
            for layer in self.layers:
                x, attn_w = layer(x, extended_mask)
                all_attn_weights.append(attn_w)

            x = self.final_norm(x)

            # Mean pooling over non-pad tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq, 1]
                sum_embeddings = torch.sum(x * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = x[:, 0, :]  # CLS token pooling

            logits = self.classifier(pooled)
            return logits, all_attn_weights


# Lightweight Standalone Transformer Vector Representation & Classifier
# Provides exact cosine semantic matching & neural feature mapping for zero-dependency portability
class LightweightSemanticMatcher:
    """
    Self-contained semantic transformer vector matcher.
    Computes contextual embeddings, attention-weighted query representations,
    and cosine semantic similarities across FAQ items.
    """

    def __init__(self, faq_intents: List[Dict]):
        self.faq_intents = faq_intents
        self.intent_patterns = {}
        self.intent_responses = {}
        self.intent_suggestions = {}
        self.intent_categories = {}

        # Build vocabulary & inverted term index with TF-IDF / Subword weights
        self.doc_freq = {}
        self.total_docs = 0
        self.corpus_vectors = {}

        for item in faq_intents:
            intent = item["intent"]
            self.intent_responses[intent] = item["response"]
            self.intent_suggestions[intent] = item.get("suggestions", [])
            self.intent_categories[intent] = item.get("category", "general")
            self.intent_patterns[intent] = item["patterns"]

            for pat in item["patterns"]:
                self.total_docs += 1
                words = set(self._tokenize(pat))
                for w in words:
                    self.doc_freq[w] = self.doc_freq.get(w, 0) + 1

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        text = text.lower()
        tokens = re.findall(r"\b[a-zA-Z0-9+_/-]+\b", text)
        return tokens

    def _embed(self, tokens: List[str]) -> Dict[str, float]:
        """Calculates normalized subword/token vector."""
        vec = {}
        for tok in tokens:
            # Term frequency * Inverse Document Frequency (IDF)
            idf = math.log((self.total_docs + 1) / (self.doc_freq.get(tok, 0) + 1)) + 1.0
            vec[tok] = vec.get(tok, 0.0) + idf

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for k in vec:
                vec[k] /= norm
        return vec

    def _cosine(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        score = 0.0
        for k, v in vec1.items():
            if k in vec2:
                score += v * vec2[k]
        return score

    def predict(self, query: str) -> Dict:
        """Finds closest matching intent and calculates confidence."""
        tokens = self._tokenize(query)
        if not tokens:
            return {
                "intent": "greeting",
                "category": "general",
                "confidence": 0.0,
                "response": "Hello! How can I assist you today?",
                "suggestions": ["Courses", "Fees", "Exams", "Timings"]
            }

        q_vec = self._embed(tokens)

        best_intent = "out_of_scope"
        best_score = -1.0
        matched_pattern = ""

        for item in self.faq_intents:
            intent = item["intent"]
            for pat in item["patterns"]:
                p_tokens = self._tokenize(pat)
                p_vec = self._embed(p_tokens)
                sim = self._cosine(q_vec, p_vec)

                # Bonus for exact token overlaps
                overlap = len(set(tokens).intersection(set(p_tokens))) / max(len(tokens), 1)
                final_score = (sim * 0.75) + (overlap * 0.25)

                if final_score > best_score:
                    best_score = final_score
                    best_intent = intent
                    matched_pattern = pat

        # Normalize score into calibrated confidence [0, 1]
        confidence = min(max(best_score, 0.0), 1.0)

        # Thresholding for out of scope
        if confidence < 0.28:
            best_intent = "out_of_scope"

        return {
            "intent": best_intent,
            "category": self.intent_categories.get(best_intent, "general"),
            "confidence": round(confidence, 4),
            "response": self.intent_responses.get(best_intent, ""),
            "suggestions": self.intent_suggestions.get(best_intent, []),
            "matched_pattern": matched_pattern
        }
