# src/core/metrics.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Set
import spacy
import logging
import re

logger = logging.getLogger(__name__)

class MetricsCalculator:
    def __init__(self):
        self.semantic_model = None
        self.ner_model = None
        self.stop = None
        self._load_semantic_model()
        self._load_ner_model()

    def _load_semantic_model(self):
        if self.semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.error("sentence_transformers not installed; semantic similarity disabled")
                self.semantic_model = False

    def _load_ner_model(self):
        if self.ner_model is None:
            try:
                import spacy
                self.ner_model = spacy.load("en_core_web_sm")
                self.stop = self.ner_model.Defaults.stop_words
            except (ImportError, OSError):
                logger.error("spacy or en_core_web_sm not installed; NER disabled")
                self.ner_model = False
                self.stop = set()

    def _tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        toks = re.findall(r"[A-Za-z0-9']+", text.lower())
        if self.stop:
            toks = [t for t in toks if t not in self.stop]
        return toks

    def lexical_recall(self, ref: str, gen: str) -> float:
        r = set(self._tokenize(ref))
        g = set(self._tokenize(gen))
        if not r:
            return 1.0 if not g else 0.0
        return len(r & g) / len(r)

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        if not self.semantic_model:
            return 0.0  # Disabled dep
        if not isinstance(text1, str) or not text1.strip() or not isinstance(text2, str) or not text2.strip():
            return 0.0
        embeddings = self.semantic_model.encode([text1, text2])
        sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return float(sim)

    def extract_entities(self, text: str) -> Set[Tuple[str, str]]:
        if not self.ner_model or not isinstance(text, str) or not text.strip():
            return set()
        doc = self.ner_model(text)
        return {(ent.text.strip(), ent.label_) for ent in doc.ents}

    def calculate_factual_recall(self, reference: str, generated: str) -> float:
        ref_entities = self.extract_entities(reference)
        gen_entities = self.extract_entities(generated)
        if not ref_entities:
            return 1.0 if not gen_entities else 0.0
        inter = ref_entities.intersection(gen_entities)
        return len(inter) / len(ref_entities)

    def is_perfect_match(self, text1: str, text2: str) -> bool:
        return text1.strip() == text2.strip() if isinstance(text1, str) and isinstance(text2, str) else False

