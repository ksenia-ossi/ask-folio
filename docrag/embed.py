"""Sentence-transformer embedding helpers."""

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

for logger_name in (
    "huggingface_hub",
    "huggingface_hub.utils._headers",
    "sentence_transformers",
    "transformers",
    "transformers.modeling_utils",
):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import numpy as np
from sentence_transformers import SentenceTransformer

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class Embedder:
    """Create normalized multilingual document and query vectors."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.e5 = "e5" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents as normalized float32 rows."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        values = self.model.encode(
            [f"passage: {text}" for text in texts] if self.e5 else texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query as a normalized float32 vector."""
        value = self.model.encode(
            f"query: {text}" if self.e5 else text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(value, dtype=np.float32).reshape(-1)
