import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from logger_config import logger

# Global storage
_docs = []
_embeddings = None
_model = None


def load_rag_data(sbert_model, csv_path="data/history_dataset.csv"):
    """
    Build embeddings from your dataset once at startup.
    """
    global _docs, _embeddings, _model

    _model = sbert_model

    logger.info("Loading RAG dataset...")

    df = pd.read_csv(csv_path, encoding="latin1")

    # Combine useful fields into a single searchable text
    texts = (
        df["chapter"].fillna("") + ". " +
        df["Grade/Topic"].fillna("") + ". " +
        df["original_text"].fillna("")
    ).tolist()

    _docs = texts

    logger.info("Creating embeddings for RAG corpus...")

    _embeddings = _model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    logger.info(f"RAG ready with {len(_docs)} documents")


def retrieve_context(query: str, top_k: int = 3):
    """
    Returns top-k most relevant history context.
    """
    global _embeddings, _docs, _model

    if _embeddings is None or len(_docs) == 0:
        return ""

    query_vec = _model.encode([query], convert_to_numpy=True)

    scores = cosine_similarity(query_vec, _embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = []
    for idx in top_indices:
        retrieved.append(_docs[idx])

    context = "\n\n".join(retrieved)

    context = context[:1200]

    return context