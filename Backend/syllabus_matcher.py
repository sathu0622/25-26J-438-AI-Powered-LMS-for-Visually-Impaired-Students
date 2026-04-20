import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    EMBEDDING_STACK_AVAILABLE = True
except Exception:
    torch = None
    F = None
    AutoModel = None
    AutoTokenizer = None
    EMBEDDING_STACK_AVAILABLE = False


EXPECTED_COLUMNS = ("chapter", "Grade/Topic", "original_text")


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> Set[str]:
    normalized = _normalize(text)
    if not normalized:
        return set()
    return {token for token in normalized.split(" ") if len(token) > 2}


@dataclass
class SyllabusRow:
    chapter: str
    grade_topic: str
    original_text: str
    tokens: Set[str]
    rag_text: str
    embedding: Optional[torch.Tensor] = None


class SyllabusMatcher:
    def __init__(self, excel_path: Path):
        self.excel_path = excel_path
        self.rows: List[SyllabusRow] = []
        self.error: Optional[str] = None
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = None
        self.model = None
        self.rag_enabled = False
        self._load()
        self._load_embedding_model()
        self._build_rag_index()

    def _load(self) -> None:
        try:
            df = pd.read_excel(self.excel_path)
            missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            if missing:
                self.error = f"Missing expected column(s): {missing}"
                return

            cleaned = df[list(EXPECTED_COLUMNS)].fillna("")
            for _, row in cleaned.iterrows():
                chapter = str(row["chapter"]).strip()
                grade_topic = str(row["Grade/Topic"]).strip()
                original_text = str(row["original_text"]).strip()
                combined = f"{chapter} {grade_topic} {original_text}"
                tokens = _tokenize(combined)
                if not tokens:
                    continue
                self.rows.append(
                    SyllabusRow(
                        chapter=chapter,
                        grade_topic=grade_topic,
                        original_text=original_text,
                        tokens=tokens,
                        rag_text=combined,
                    )
                )
            if not self.rows:
                self.error = "No usable rows found in syllabus Excel."
        except Exception as exc:
            self.error = str(exc)

    def _load_embedding_model(self) -> None:
        if self.error:
            return
        if not EMBEDDING_STACK_AVAILABLE:
            self.rag_enabled = False
            print("Embedding stack unavailable, using token fallback.")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.model = AutoModel.from_pretrained(self.embedding_model_name)
            self.model.eval()
            self.rag_enabled = True
        except Exception as exc:
            self.rag_enabled = False
            print(f"RAG embedding model unavailable, using token fallback: {exc}")

    def _mean_pool(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if torch is None:
            raise RuntimeError("Torch is unavailable for mean pooling.")
        token_embeddings = model_output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode_text(self, text: str) -> Optional[torch.Tensor]:
        if not self.rag_enabled or not self.tokenizer or not self.model:
            return None
        try:
            encoded = self.tokenizer(
                text or "",
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**encoded)
            pooled = self._mean_pool(output, encoded["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            return normalized.squeeze(0)
        except Exception as exc:
            self.rag_enabled = False
            print(f"RAG encoding failed, switching to fallback matching: {exc}")
            return None

    def _build_rag_index(self) -> None:
        if not self.rag_enabled:
            return
        for row in self.rows:
            row.embedding = self._encode_text(row.rag_text)

    def _semantic_retrieve(self, article_text: str, top_k: int = 3) -> List[Tuple[float, SyllabusRow]]:
        query_embedding = self._encode_text(article_text)
        if query_embedding is None:
            return []

        scored: List[Tuple[float, SyllabusRow]] = []
        for row in self.rows:
            if row.embedding is None:
                continue
            score = float(torch.dot(query_embedding, row.embedding).item())
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:max(top_k, 1)]

    def match_article(
        self,
        article_text: str,
        article_heading: str = "",
        threshold: float = 0.12,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        if self.error:
            return {"error": f"Syllabus matcher not ready: {self.error}"}

        retrieval_query = f"{article_heading} {article_text}".strip()
        article_tokens = _tokenize(retrieval_query)
        if not article_tokens:
            return {"error": "Article text is empty or invalid for syllabus matching."}

        # RAG Retrieval step (semantic). Falls back to token matching if model unavailable.
        semantic_top = self._semantic_retrieve(retrieval_query, top_k=max(top_k * 2, 6))

        if semantic_top:
            rescored: List[Tuple[float, SyllabusRow]] = []
            for semantic_score, row in semantic_top:
                overlap = article_tokens.intersection(row.tokens)
                lexical_signal = len(overlap) / max(len(row.tokens), 1) if overlap else 0.0
                final_score = (0.8 * semantic_score) + (0.2 * lexical_signal)
                rescored.append((final_score, row))
            rescored.sort(key=lambda x: x[0], reverse=True)
            top = rescored[:max(top_k, 1)]
            method = "rag_semantic_retrieval"
        else:
            fallback_scored: List[Tuple[float, SyllabusRow]] = []
            for row in self.rows:
                overlap = article_tokens.intersection(row.tokens)
                if not overlap:
                    continue
                precision_like = len(overlap) / max(len(row.tokens), 1)
                recall_like = len(overlap) / max(len(article_tokens), 1)
                score = (0.75 * precision_like) + (0.25 * recall_like)
                fallback_scored.append((score, row))
            fallback_scored.sort(key=lambda x: x[0], reverse=True)
            top = fallback_scored[:max(top_k, 1)]
            method = "token_overlap_fallback"

        if not top:
            return {
                "in_syllabus": False,
                "confidence": 0.0,
                "match": None,
                "alternatives": [],
                "method": method,
            }

        best_score, best_row = top[0]
        in_syllabus = best_score >= threshold
        alternatives = [
            {
                "chapter": row.chapter,
                "grade_topic": row.grade_topic,
                "confidence": round(score, 4),
            }
            for score, row in top[1:]
        ]

        return {
            "in_syllabus": in_syllabus,
            "confidence": round(best_score, 4),
            "match": {
                "chapter": best_row.chapter,
                "grade_topic": best_row.grade_topic,
                "original_text": best_row.original_text,
            },
            "alternatives": alternatives,
            "method": method,
        }
