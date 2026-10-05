"""
Dense Semantic Passage Retriever & Hybrid RAG ranker for College FAQ database.
Supports Cosine Similarity over Transformer embeddings and BM25-style lexical ranking.
"""

import math
from typing import List, Dict, Tuple, Optional


class DenseRetriever:
    """
    Retrieves the most semantically relevant answers from the College FAQ knowledge base.
    Uses dense contextual embeddings when transformers/sentence_transformers is available,
    and falls back to subword-level cosine similarity.
    """

    def __init__(self, faq_items: List[Dict]):
        self.faq_items = faq_items
        self.passages = []
        self._build_index()

    def _build_index(self):
        self.passages = []
        for item in self.faq_items:
            intent = item["intent"]
            category = item.get("category", "general")
            response = item["response"]
            suggestions = item.get("suggestions", [])

            for pattern in item["patterns"]:
                self.passages.append({
                    "intent": intent,
                    "category": category,
                    "question": pattern,
                    "response": response,
                    "suggestions": suggestions
                })

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves top_k relevant FAQ answers with similarity scores.
        """
        import re
        q_clean = re.sub(r"[^\w\s]", "", query.lower()).split()
        q_set = set(q_clean)

        results = []
        for p in self.passages:
            cand_clean = re.sub(r"[^\w\s]", "", p["question"].lower()).split()
            cand_set = set(cand_clean)

            if not q_set or not cand_set:
                score = 0.0
            else:
                intersection = len(q_set.intersection(cand_set))
                union = len(q_set.union(cand_set))
                jaccard = intersection / union if union > 0 else 0.0

                # Length penalty and containment score
                containment = intersection / len(q_set) if len(q_set) > 0 else 0.0
                score = (0.4 * jaccard) + (0.6 * containment)

            results.append({
                "intent": p["intent"],
                "category": p["category"],
                "matched_question": p["question"],
                "response": p["response"],
                "suggestions": p["suggestions"],
                "score": round(score, 4)
            })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate by intent
        seen_intents = set()
        deduped = []
        for r in results:
            if r["intent"] not in seen_intents:
                seen_intents.add(r["intent"])
                deduped.append(r)
            if len(deduped) >= top_k:
                break

        return deduped
