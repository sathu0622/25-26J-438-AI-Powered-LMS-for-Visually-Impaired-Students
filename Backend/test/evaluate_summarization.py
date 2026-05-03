"""
Evaluate summarization quality from a JSON dataset.

Usage:
    python evaluate_summarization.py --dataset sample_summarization_eval.json
    python evaluate_summarization.py --dataset my_eval.json --save-report eval_report.json
    python evaluate_summarization.py --dataset my_eval.json --use-json-output-only
    python evaluate_summarization.py --dataset my_eval.json --limit 3

Dataset format:
{
  "samples": [
    {
      "id": "news_001",
      "resource_type": "newspapers",
      "input_text": "...",
      "expected_summary": "...",
      "model_output": "..."   // optional; used directly if present
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from models import load_all_models
from summarizer import summarize_text


def normalize_resource_type(value: str) -> str:
    text = (value or "").strip().lower()
    mapping = {
        "newspaper": "newspapers",
        "newspapers": "newspapers",
        "magazine": "magazine",
        "magazines": "magazine",
        "book": "books",
        "books": "books",
    }
    return mapping.get(text, text)


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    rows, cols = len(a) + 1, len(b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def token_overlap_metrics(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())

    precision = safe_div(overlap, len(pred_tokens))
    recall = safe_div(overlap, len(ref_tokens))
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = safe_div(lcs, len(pred_tokens))
    recall = safe_div(lcs, len(ref_tokens))
    return safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0


def sentence_count(text: str) -> int:
    parts = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    return len(parts)


def evaluate_sample(sample: Dict[str, str], prediction: str) -> Dict[str, float]:
    expected_summary = sample["expected_summary"]
    input_text = sample["input_text"]

    overlap = token_overlap_metrics(prediction, expected_summary)
    rouge_l = rouge_l_f1(prediction, expected_summary)

    pred_words = len(tokenize(prediction))
    ref_words = len(tokenize(expected_summary))
    input_words = len(tokenize(input_text))

    return {
        **overlap,
        "rouge_l_f1": rouge_l,
        "pred_words": float(pred_words),
        "ref_words": float(ref_words),
        "input_words": float(input_words),
        "length_ratio_pred_to_ref": safe_div(pred_words, ref_words),
        "compression_ratio_input_to_pred": safe_div(input_words, pred_words),
        "pred_sentence_count": float(sentence_count(prediction)),
        "ref_sentence_count": float(sentence_count(expected_summary)),
    }


def format_score(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else "nan"


def summarize_group(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "token_precision",
        "token_recall",
        "token_f1",
        "rouge_l_f1",
        "length_ratio_pred_to_ref",
        "compression_ratio_input_to_pred",
        "pred_words",
        "ref_words",
        "pred_sentence_count",
        "ref_sentence_count",
    ]
    return {key: mean(r[key] for r in rows) if rows else 0.0 for key in keys}


def load_dataset(dataset_path: Path) -> List[Dict[str, str]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError("Dataset JSON must include a non-empty 'samples' list.")

    required_fields = {"resource_type", "input_text", "expected_summary"}
    for idx, sample in enumerate(samples, start=1):
        missing = [field for field in required_fields if not str(sample.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Sample #{idx} missing required fields: {missing}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate summarization model with rough metrics.")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset.")
    parser.add_argument("--save-report", default="", help="Optional path to save full JSON report.")
    parser.add_argument(
        "--use-json-output-only",
        action="store_true",
        help="Use sample.model_output only; do not run model generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Only evaluate the first N samples (0 = all). Useful for quick CPU smoke tests.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples = load_dataset(dataset_path)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
        print(f"Using first {len(samples)} sample(s) (--limit {args.limit}).", flush=True)

    models = None
    if not args.use_json_output_only:
        print("Loading summarization model...")
        models = load_all_models()
        print("Model loaded.\n", flush=True)

    evaluated_rows: List[Dict[str, float]] = []
    by_type_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    detailed_results = []

    total = len(samples)
    if not args.use_json_output_only:
        print(
            f"Evaluating {total} sample(s). On CPU, each summary can take 1–5+ minutes "
            f"(beam search + sampling); progress prints below.\n",
            flush=True,
        )

    for idx, sample in enumerate(samples, start=1):
        resource_type = normalize_resource_type(sample["resource_type"])
        input_text = sample["input_text"]
        sid = sample.get("id", f"sample_{idx}")

        prediction = (sample.get("model_output") or "").strip()
        if not prediction:
            if args.use_json_output_only:
                raise ValueError(
                    f"Sample #{idx} has no model_output while --use-json-output-only is enabled."
                )
            print(
                f"[{idx}/{total}] {sid} ({resource_type}) — generating summary... ",
                end="",
                flush=True,
            )
            t0 = time.perf_counter()
            prediction = summarize_text(
                input_text,
                resource_type,
                models["summ_tokenizer"],
                models["summ_model"],
            )
            print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)
        else:
            print(f"[{idx}/{total}] {sid} — using model_output from JSON", flush=True)

        metrics = evaluate_sample(sample, prediction)
        evaluated_rows.append(metrics)
        by_type_rows[resource_type].append(metrics)

        detailed_results.append(
            {
                "id": sample.get("id", f"sample_{idx}"),
                "resource_type": resource_type,
                "prediction": prediction,
                "expected_summary": sample["expected_summary"],
                "metrics": metrics,
            }
        )

    overall = summarize_group(evaluated_rows)
    by_type = {k: summarize_group(v) for k, v in by_type_rows.items()}

    print("=" * 70)
    print("Summarization Evaluation (Rough Metrics)")
    print("=" * 70)
    print(f"Total samples: {len(samples)}")
    print("")

    print("Overall")
    print("-" * 70)
    print(f"Token F1          : {format_score(overall['token_f1'])}")
    print(f"ROUGE-L F1        : {format_score(overall['rouge_l_f1'])}")
    print(f"Token Precision   : {format_score(overall['token_precision'])}")
    print(f"Token Recall      : {format_score(overall['token_recall'])}")
    print(f"Length Ratio P/R  : {format_score(overall['length_ratio_pred_to_ref'])}")
    print(f"Compression I/P   : {format_score(overall['compression_ratio_input_to_pred'])}")
    print("")

    print("By Resource Type")
    print("-" * 70)
    for resource_type in sorted(by_type.keys()):
        m = by_type[resource_type]
        print(
            f"{resource_type:12} | "
            f"TokenF1={format_score(m['token_f1'])} | "
            f"ROUGE-L={format_score(m['rouge_l_f1'])} | "
            f"Len(P/R)={format_score(m['length_ratio_pred_to_ref'])} | "
            f"Comp(I/P)={format_score(m['compression_ratio_input_to_pred'])}"
        )

    if args.save_report:
        report = {
            "dataset": str(dataset_path),
            "total_samples": len(samples),
            "overall": overall,
            "by_resource_type": by_type,
            "detailed_results": detailed_results,
        }
        save_path = Path(args.save_report)
        save_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("")
        print(f"Saved full report to: {save_path}")


if __name__ == "__main__":
    main()
