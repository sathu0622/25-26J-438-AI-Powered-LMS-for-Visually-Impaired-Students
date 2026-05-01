import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = ("chapter", "Grade/Topic", "original_text")

# Minimum overlap ratio to consider content as "in syllabus".
# Tune this between 0.2 – 0.4 based on your real data.
DEFAULT_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation from text."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> Set[str]:
    """Return a set of meaningful tokens (length > 2) from text."""
    normalized = _normalize(text)
    if not normalized:
        return set()
    return {token for token in normalized.split() if len(token) > 2}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SyllabusRow:
    chapter: str
    grade_topic: str
    original_text: str
    tokens: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class SyllabusMatcher:
    """
    Matches article text against syllabus entries loaded from an Excel file.

    Matching strategy:
      - Tokenise both the article and each syllabus row.
      - Score = overlap / syllabus_row_tokens  (precision-like).
      - A secondary recall signal is added with lower weight.
      - Best score >= threshold  →  in_syllabus = True.
    """

    def __init__(self, excel_path: Path, threshold: float = DEFAULT_THRESHOLD):
        self.excel_path = excel_path
        self.default_threshold = threshold
        self.rows: List[SyllabusRow] = []
        self.error: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load and validate the syllabus Excel file."""
        try:
            df = pd.read_excel(self.excel_path)
        except Exception as exc:
            self.error = f"Could not read Excel file: {exc}"
            return

        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            self.error = f"Missing expected column(s): {missing}"
            return

        cleaned = df[list(EXPECTED_COLUMNS)].fillna("")
        for _, row in cleaned.iterrows():
            chapter      = str(row["chapter"]).strip()
            grade_topic  = str(row["Grade/Topic"]).strip()
            original_text = str(row["original_text"]).strip()

            combined_text = f"{chapter} {grade_topic} {original_text}"
            tokens = _tokenize(combined_text)

            if not tokens:
                continue  # skip empty / whitespace-only rows

            self.rows.append(
                SyllabusRow(
                    chapter=chapter,
                    grade_topic=grade_topic,
                    original_text=original_text,
                    tokens=tokens,
                )
            )

        if not self.rows:
            self.error = "No usable rows found in the syllabus Excel file."

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_article(
        self,
        article_text: str,
        article_heading: str = "",
        threshold: Optional[float] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Check whether article content belongs to any syllabus entry.

        Parameters
        ----------
        article_text    : Main body of the article.
        article_heading : Optional heading / title (improves matching).
        threshold       : Override the instance-level default threshold.
        top_k           : How many alternative matches to return.

        Returns
        -------
        dict with keys:
            in_syllabus  (bool)
            confidence   (float 0–1)
            match        (dict | None)  – best matching row
            alternatives (list[dict])   – next best matches
            message      (str)
        """
        if self.error:
            return {"error": f"SyllabusMatcher not ready: {self.error}"}

        effective_threshold = threshold if threshold is not None else self.default_threshold

        # Build article token set from heading + body
        query = f"{article_heading} {article_text}".strip()
        article_tokens = _tokenize(query)

        if not article_tokens:
            return {"error": "Article text is empty or could not be tokenised."}

        # Score every syllabus row
        scored: List[tuple] = []
        for row in self.rows:
            overlap = article_tokens.intersection(row.tokens)
            if not overlap:
                continue

            # Precision-like: how much of the syllabus topic is covered?
            precision = len(overlap) / max(len(row.tokens), 1)
            # Recall-like: how much of the article maps to the topic?
            recall = len(overlap) / max(len(article_tokens), 1)

            # Weighted combination – precision matters more for a yes/no check
            score = (0.75 * precision) + (0.25 * recall)
            scored.append((score, row))

        if not scored:
            return {
                "in_syllabus": False,
                "confidence": 0.0,
                "match": None,
                "alternatives": [],
                "message": "Content is not under the subject.",
            }

        # Sort descending and pick top results
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_row = scored[0]
        alternatives = [
            {
                "chapter": row.chapter,
                "grade_topic": row.grade_topic,
                "confidence": round(score, 4),
            }
            for score, row in scored[1:top_k]
        ]

        in_syllabus = best_score >= effective_threshold

        return {
            "in_syllabus": in_syllabus,
            "confidence": round(best_score, 4),
            "match": {
                "chapter": best_row.chapter,
                "grade_topic": best_row.grade_topic,
                "original_text": best_row.original_text,
            } if in_syllabus else None,
            "alternatives": alternatives,
            "message": (
                "Content is under this subject."
                if in_syllabus
                else "Content is not under the subject."
            ),
        }