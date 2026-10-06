"""
Core College Enquiry Chatbot Engine.
Integrates Transformer intent classification, slot/entity extraction,
dialogue state tracking, confidence calibration, and contextual response dispatching.
"""

import os
import re
import json
from typing import Dict, List, Optional, Any

from src.tokenizer import SimpleTokenizer
from models.custom_transformer import LightweightSemanticMatcher
from models.dense_retriever import DenseRetriever


class CollegeChatbotEngine:
    """
    Main Chatbot Orchestration Engine.
    Handles user queries across 6 core college domains:
    - Courses (UG, PG, eligibility, curriculum, duration)
    - Fees (Tuition, hostel, exams, scholarships, payment modes)
    - Exams (Datesheets, hall tickets, grading, revaluation, backlogs)
    - Departments (CSE, ECE, Mech, Civil, Management, Labs)
    - Timings (College hours, library, office, gym & sports)
    - Facilities (Hostel, library, Wi-Fi, transport, food court, sports, health center)
    """

    CONFIDENCE_THRESHOLD = 0.32

    def __init__(
        self,
        data_path: Optional[str] = None,
        entity_rules_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if data_path is None:
            data_path = os.path.join(base_dir, "data", "college_faq_dataset.json")
        if entity_rules_path is None:
            entity_rules_path = os.path.join(base_dir, "data", "entity_rules.json")

        self.data_path = data_path
        self.entity_rules_path = entity_rules_path
        self.checkpoint_path = checkpoint_path

        # 1. Load Knowledge Base
        with open(data_path, "r", encoding="utf-8") as f:
            self.faq_db = json.load(f)

        self.college_name = self.faq_db.get("college_name", "Apex Institute of Science & Technology")
        self.categories = self.faq_db.get("categories", [])
        self.intents = self.faq_db["intents"]

        # Map intent details
        self.intent_map = {item["intent"]: item for item in self.intents}

        # 2. Load Entity Rules
        self.entity_rules = {}
        if os.path.exists(entity_rules_path):
            with open(entity_rules_path, "r", encoding="utf-8") as f:
                self.entity_rules = json.load(f).get("entities", {})

        # 3. Initialize Matcher & Dense Retriever
        self.matcher = LightweightSemanticMatcher(self.intents)
        self.retriever = DenseRetriever(self.intents)

        # 4. Multi-turn Session Memory
        self.dialogue_history: List[Dict[str, Any]] = []
        self.current_context: Optional[str] = None

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extracts college domain entities and slots from text."""
        extracted = {}
        q_lower = query.lower()

        for entity_type, keywords in self.entity_rules.items():
            found = []
            for kw in keywords:
                # Word boundary match
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                if re.search(pattern, q_lower):
                    found.append(kw)
            if found:
                extracted[entity_type] = found

        return extracted

    def get_fallback_response(self, query: str) -> Dict[str, Any]:
        """Handles low confidence / out-of-scope queries gracefully."""
        return {
            "query": query,
            "intent": "out_of_scope",
            "category": "out_of_scope",
            "confidence": 0.0,
            "entities": {},
            "response": (
                f"I'm sorry, I couldn't find a direct match for your question regarding {self.college_name}. "
                "I specialize in answering questions about Courses, Fees, Exams, Departments, Timings, and Campus Facilities. "
                "Could you please rephrase or choose one of the options below?"
            ),
            "suggestions": [
                "Undergraduate courses",
                "Tuition fee structure",
                "Exam schedule",
                "Hostel facilities",
                "Contact admission helpline"
            ]
        }

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Full processing pipeline for an incoming user query:
        1. Preprocessing & entity extraction
        2. Intent classification & confidence estimation
        3. Contextual dialogue resolution
        4. Response formulation with suggested follow-up chips
        """
        clean_query = user_query.strip()
        if not clean_query:
            return {
                "query": "",
                "intent": "greeting",
                "category": "general",
                "confidence": 1.0,
                "entities": {},
                "response": f"Hello! Welcome to the {self.college_name} Enquiry Chatbot. How may I help you today?",
                "suggestions": ["Courses offered", "Fee structure", "Exam dates", "Hostel facilities"]
            }

        # 1. Entity Extraction
        entities = self.extract_entities(clean_query)

        # 2. Intent Prediction
        pred = self.matcher.predict(clean_query)
        intent_name = pred["intent"]
        confidence = pred["confidence"]
        category = pred["category"]

        # 3. Contextual state update / resolution
        if self.current_context and confidence < 0.45:
            # Check if query is a contextual follow-up (e.g., "what about fees?" or "and timings?")
            if "fee" in clean_query.lower() and self.current_context.startswith("course_"):
                intent_name = "fee_structure"
                confidence = 0.85
            elif "hostel" in clean_query.lower():
                intent_name = "hostel_facility"
                confidence = 0.85

        # 4. Out-of-Scope / Confidence check
        if confidence < self.CONFIDENCE_THRESHOLD or intent_name == "out_of_scope":
            # Attempt dense retrieval as backup
            dense_hits = self.retriever.retrieve(clean_query, top_k=1)
            if dense_hits and dense_hits[0]["score"] > 0.40:
                best_hit = dense_hits[0]
                intent_name = best_hit["intent"]
                category = best_hit["category"]
                confidence = best_hit["score"]
                item = self.intent_map[intent_name]
                response_text = item["response"]
                suggestions = item.get("suggestions", [])
            else:
                return self.get_fallback_response(clean_query)
        else:
            item = self.intent_map.get(intent_name, {})
            response_text = item.get("response", "")
            suggestions = item.get("suggestions", [])

        # Update context
        self.current_context = intent_name
        result = {
            "query": clean_query,
            "intent": intent_name,
            "category": category,
            "confidence": confidence,
            "entities": entities,
            "response": response_text,
            "suggestions": suggestions
        }

        # Log turn in history
        self.dialogue_history.append({
            "user": clean_query,
            "bot": response_text,
            "intent": intent_name
        })

        return result

    def reset_conversation(self):
        """Clears dialog history and context."""
        self.dialogue_history = []
        self.current_context = None
