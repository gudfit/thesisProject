from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Set
import spacy
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:

    def __init__(self):
        self.semantic_model = None
        self.ner_model = None

    def _load_semantic_model(self):
        if self.semantic_model is None:
            logger.info("Loading sentence transformer model...")
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_ner_model(self):
        if self.ner_model is None:
            try:
                logger.info("Loading spaCy NER model...")
                self.ner_model = spacy.load("en_core_web_sm")
            except IOError:
                logger.warning("spaCy model not found. NER features disabled.")
                self.ner_model = False

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        self._load_semantic_model()
        embeddings = self.semantic_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def extract_entities(self, text: str) -> Set[Tuple[str, str]]:
        self._load_ner_model()

        if not self.ner_model or not isinstance(text, str) or not text.strip():
            return set()

        doc = self.ner_model(text)
        return {(ent.text.strip(), ent.label_) for ent in doc.ents}

    def calculate_factual_recall(self, reference: str, generated: str) -> float:
        ref_entities = self.extract_entities(reference)
        gen_entities = self.extract_entities(generated)
        if not ref_entities:
            return 1.0 if not gen_entities else 0.0

        intersection = ref_entities.intersection(gen_entities)
        return len(intersection) / len(ref_entities)

    def is_perfect_match(self, text1: str, text2: str) -> bool:
        return text1.strip() == text2.strip()
