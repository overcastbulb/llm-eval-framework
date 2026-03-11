import os
import warnings
import logging
from dotenv import load_dotenv

load_dotenv()

# Suppress all warnings before importing sentence_transformers
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN", "")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("FlagEmbedding").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer, util


class AccuracyEvaluator:
    """
    Scores how accurate the LLM answer is compared to the expected answer.
    Uses two methods:
    - Exact match: checks if expected answer appears in the response
    - Semantic match: uses sentence embeddings to measure meaning similarity
    """

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

    def evaluate(self, expected: str, actual: str) -> dict:
        # Exact match — simple substring check
        exact = expected.lower().strip() in actual.lower().strip()

        # Semantic similarity — cosine similarity of embeddings
        emb_expected = self.model.encode(expected, normalize_embeddings=True)
        emb_actual = self.model.encode(actual, normalize_embeddings=True)
        semantic_score = float(util.cos_sim(emb_expected, emb_actual))

        # Combined score (weighted average)
        combined = 0.4 * int(exact) + 0.6 * semantic_score

        return {
            "exact_match": exact,
            "semantic_score": round(semantic_score, 4),
            "accuracy_score": round(combined, 4)
        }