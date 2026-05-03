import requests
import json
import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_filename = LOG_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("EVAL_API_URL", "http://localhost:8000/evaluate")
INPUT_FILE = Path(os.getenv("EVAL_INPUT", "test_questions.json"))
OUTPUT_FILE = Path(os.getenv("EVAL_OUTPUT", "test_results.json"))
REQUEST_TIMEOUT = int(os.getenv("EVAL_TIMEOUT", "300"))


# ── Load test cases ───────────────────────────────────────────────────────────
def load_test_cases(path: Path) -> list[dict]:
    log.info(f"Loading test cases from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("test_cases", [])
    if not cases:
        log.error("No test_cases found in JSON.")
        sys.exit(1)

    log.info(f"Loaded {len(cases)} test case(s).")
    return cases


# ── API call ──────────────────────────────────────────────────────────────────
def call_evaluate(question: str, student_answer: str) -> dict:
    payload = {
        "question": question,
        "student_answer": student_answer
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()
    return response.json()


# ── Run tests ─────────────────────────────────────────────────────────────────
def run_tests(cases: list[dict]) -> list[dict]:
    results = []

    for tc in cases:
        tc_id = tc.get("id", "?")
        subject = tc.get("subject", "N/A")
        question = tc.get("question", "").strip()
        std_answer = tc.get("student_answer", "").strip()

        log.info("─" * 60)
        log.info(f"[TC {tc_id}] Subject  : {subject}")
        log.info(f"[TC {tc_id}] Question : {question}")
        log.info(f"[TC {tc_id}] Student  : {std_answer}")

        try:
            api_resp = call_evaluate(question, std_answer)

            model_answer = api_resp.get("model_answer", "N/A")

            # 🔥 FIXED: correct key usage
            score = api_resp.get("final_score")

            semantic = api_resp.get("semantic_similarity")
            keyword = api_resp.get("keyword_match")
            jaccard = api_resp.get("jaccard_similarity")

            log.info(f"[TC {tc_id}] Model Answer : {model_answer}")
            log.info(f"[TC {tc_id}] Score        : {score}")

            result = {
                **tc,
                "model_answer": model_answer,
                "final_score": score,
                "semantic_similarity": semantic,
                "keyword_match": keyword,
                "jaccard_similarity": jaccard,
                "status": "success",
                "raw_response": api_resp,
                "evaluated_at": datetime.now(timezone.utc).isoformat()
            }

        except requests.exceptions.ConnectionError:
            log.error(f"[TC {tc_id}] Connection error.")
            result = {**tc, "final_score": None, "status": "connection_error"}

        except requests.exceptions.Timeout:
            log.error(f"[TC {tc_id}] Timeout after {REQUEST_TIMEOUT}s.")
            result = {**tc, "final_score": None, "status": "timeout"}

        except requests.exceptions.HTTPError as e:
            log.error(f"[TC {tc_id}] HTTP error: {e}")
            result = {**tc, "final_score": None, "status": "http_error"}

        except Exception as e:
            log.exception(f"[TC {tc_id}] Unexpected error: {e}")
            result = {**tc, "final_score": None, "status": "error"}

        results.append(result)

    return results


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(results: list[dict], path: Path) -> None:
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "results": results
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"Results saved to: {path}")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(results: list[dict]) -> None:
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    log.info(f"Total   : {len(results)}")
    log.info(f"Success : {len(success)}")
    log.info(f"Failed  : {len(failed)}")

    scores = [r["final_score"] for r in success if r.get("final_score") is not None]

    if scores:
        avg = sum(scores) / len(scores)
        log.info(f"Avg Score : {avg:.2f} | Min: {min(scores):.2f} | Max: {max(scores):.2f}")

    if failed:
        log.warning("Failed test cases:")
        for r in failed:
            log.warning(f"  TC {r.get('id')} — {r['status']}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"Starting evaluation run — API: {API_URL}")
    log.info(f"Log file: {log_filename}")

    cases = load_test_cases(INPUT_FILE)
    results = run_tests(cases)

    save_results(results, OUTPUT_FILE)
    print_summary(results)

    log.info("Done.")