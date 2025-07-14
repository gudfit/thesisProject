# src/metrics.py
import torch
from sentence_transformers import SentenceTransformer, util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SBERT_MODEL = None 

try:
    SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    print(f"SBERT model 'all-MiniLM-L6-v2' loaded onto {DEVICE}.")
except Exception as e:
    print(f"Warning: Could not load SBERT model. Semantic metrics will be disabled.")
    print(f"Error details: {e}")

def calculate_semantic_similarity(sentence1: str, sentence2: str) -> float:
    if SBERT_MODEL is None or not isinstance(sentence1, str) or not isinstance(sentence2, str):
        return -1.0

    try:
        embeddings = SBERT_MODEL.encode([sentence1, sentence2], convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings[0], embeddings[1])
        return cosine_score.item()
    except Exception as e:
        print(f"Error during semantic similarity calculation: {e}")
        return -1.0
