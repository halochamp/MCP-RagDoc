"""Deterministic retrieval-quality evaluator for MCP-RagDoc.

Input JSON is a list of objects:
  {"id": "case-name", "query": "...", "relevant": ["expected-file.txt"]}

The evaluator calls the real RAG service, extracts ranked source basenames, and
reports Recall@1/3/8 plus mean reciprocal rank.  It never calls an LLM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


_SOURCE_RE = re.compile(r"^\s*\[source:\s*(.+?)\]\s*$", re.MULTILINE)
_DEFAULT_K = (1, 3, 8)


def _load_cases(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("golden file must contain a non-empty JSON list")
    cases: list[dict[str, object]] = []
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise ValueError(f"case {index} must be an object")
        query = item.get("query")
        relevant = item.get("relevant")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"case {index} query must be a non-empty string")
        if not isinstance(relevant, list) or not relevant or not all(
            isinstance(value, str) and value.strip() for value in relevant
        ):
            raise ValueError(f"case {index} relevant must be a non-empty string list")
        cases.append({
            "id": str(item.get("id") or f"case-{index}"),
            "query": query.strip(),
            "relevant": [Path(value).name.casefold() for value in relevant],
        })
    return cases


def _ranked_basenames(result: str) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for match in _SOURCE_RE.finditer(result):
        name = Path(match.group(1).strip()).name.casefold()
        if name and name not in seen:
            seen.add(name)
            ranked.append(name)
    return ranked


def score_cases(cases: list[dict[str, object]], search_fn) -> dict[str, object]:
    recalls = {k: 0 for k in _DEFAULT_K}
    reciprocal_sum = 0.0
    details: list[dict[str, object]] = []
    for case in cases:
        result = search_fn(str(case["query"]))
        ranked = _ranked_basenames(result)
        relevant = set(case["relevant"])
        first_rank = next(
            (rank for rank, name in enumerate(ranked, 1) if name in relevant),
            None,
        )
        for k in _DEFAULT_K:
            if any(name in relevant for name in ranked[:k]):
                recalls[k] += 1
        rr = 1.0 / first_rank if first_rank else 0.0
        reciprocal_sum += rr
        details.append({
            "id": case["id"],
            "query": case["query"],
            "relevant": sorted(relevant),
            "ranked": ranked[: max(_DEFAULT_K)],
            "first_relevant_rank": first_rank,
            "reciprocal_rank": round(rr, 6),
        })
    total = len(cases)
    return {
        "cases": total,
        "recall_at_1": round(recalls[1] / total, 6),
        "recall_at_3": round(recalls[3] / total, 6),
        "recall_at_8": round(recalls[8] / total, 6),
        "mrr": round(reciprocal_sum / total, 6),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MCP-RagDoc retrieval without an LLM")
    parser.add_argument("golden", type=Path, help="JSON file containing query/relevant cases")
    parser.add_argument(
        "--mode",
        choices=("hybrid", "lexical"),
        default="hybrid",
        help="hybrid enables MiniLM semantic retrieval; lexical disables it",
    )
    parser.add_argument("--details", action="store_true", help="include per-query rankings")
    parser.add_argument("--knowledge-dir", help="knowledge directory used by the evaluated instance")
    parser.add_argument("--data-dir", help="private MCP-RagDoc state directory")
    parser.add_argument("--db", help="explicit SQLite database path")
    parser.add_argument("--semantic-model-dir", help="explicit local MiniLM snapshot directory")
    args = parser.parse_args()

    os.environ["RAGDOC_SEMANTIC"] = "0" if args.mode == "lexical" else "1"
    if args.knowledge_dir:
        os.environ["RAGDOC_KNOWLEDGE_DIR"] = args.knowledge_dir
    if args.data_dir:
        os.environ["RAGDOC_DATA_DIR"] = args.data_dir
    if args.db:
        os.environ["RAGDOC_DB"] = args.db
    if args.semantic_model_dir:
        os.environ["RAGDOC_SEMANTIC_MODEL_DIR"] = args.semantic_model_dir

    # Import after applying instance configuration so config sees it.
    import rag_service

    cases = _load_cases(args.golden)
    report = score_cases(cases, rag_service.search)
    report["mode"] = args.mode
    if not args.details:
        report.pop("details", None)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
