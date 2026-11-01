"""Evaluation metrics for retrieval, answers, and citations."""

import json
import re
import statistics
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    """Load newline-delimited question records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    """Normalize common punctuation and spacing for answer comparisons."""
    text = text.lower().replace("\u2212", "-").replace("\u2013", "-")
    return re.sub(r"[\s,\u202f\u00a0€]", "", text)


def is_correct(question: dict, found: bool, answer: str) -> bool:
    """Check refusal correctness or expected answer inclusion."""
    if not question["answerable"]:
        return not found
    return found and any(normalize(expected) in normalize(answer) for expected in question["expect"])


def page_hit(question: dict, pages: list[dict]) -> bool:
    """Check whether retrieved pages intersect the gold pages."""
    gold = {(question["source"], page) for page in question["pages"]}
    return any((page["source"], page["page"]) in gold for page in pages)


def retrieval_hit_rate(questions: list[dict], retriever, k: int) -> float:
    """Calculate retrieval page hit rate over answerable questions."""
    answerable = [question for question in questions if question["answerable"]]
    if not answerable:
        return 0.0
    hits = [
        page_hit(question, [doc.metadata for doc in retriever.search(question["question"], k=k)])
        for question in answerable
    ]
    return sum(hits) / len(hits)


def supported(question: dict, cited: list[dict], page_text: dict) -> bool:
    """Check whether a citation page contains an expected answer."""
    return any(
        normalize(expected) in page_text.get((citation["source"], citation["page"]), "")
        for citation in cited
        for expected in question["expect"]
    )


def score(question: dict, result, page_text: dict) -> dict:
    """Create a per-question evaluation record."""
    row = {
        "id": question["id"],
        "question": question["question"],
        "answerable": question["answerable"],
        "found": result.found,
        "answer": result.answer,
        "sources": result.sources,
        "correct": is_correct(question, result.found, result.answer),
        "latency_s": result.latency_s,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
    if question["answerable"]:
        row["retrieval_hit"] = page_hit(question, result.retrieved)
        row["citation_ok"] = result.found and supported(question, result.sources, page_text)
    return row


def summarize(rows: list[dict]) -> dict:
    """Summarize answer quality, retrieval, latency, and token usage."""
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    latencies = [row["latency_s"] for row in rows]

    def rate(items, key):
        return round(sum(item[key] for item in items) / len(items), 3) if items else None

    return {
        "questions": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "answer_accuracy": rate(answerable, "correct"),
        "citation_accuracy": rate(answerable, "citation_ok"),
        "retrieval_hit_rate": rate(answerable, "retrieval_hit"),
        "false_refusals": sum(not row["found"] for row in answerable),
        "correct_refusals": sum(row["correct"] for row in unanswerable),
        "hallucinated_answers": sum(row["found"] for row in unanswerable),
        "latency_median_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_max_s": round(max(latencies), 2) if latencies else None,
        "input_tokens_mean": round(statistics.mean(row["input_tokens"] for row in rows)) if rows else 0,
        "output_tokens_mean": round(statistics.mean(row["output_tokens"] for row in rows)) if rows else 0,
    }
