"""Model-free tests for the deterministic MCP-RagDoc evaluator."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag_eval


def test_ranked_basenames_deduplicates_sources_in_rank_order():
    text = (
        "- x\n  [source: /kb/a.txt]\n"
        "- y\n  [source: /kb/b.txt]\n"
        "- duplicate\n  [source: /kb/a.txt]\n"
    )
    assert rag_eval._ranked_basenames(text) == ["a.txt", "b.txt"]


def test_score_cases_reports_recall_and_mrr():
    cases = [
        {"id": "one", "query": "q1", "relevant": ["a.txt"]},
        {"id": "two", "query": "q2", "relevant": ["c.txt"]},
    ]
    outputs = {
        "q1": "[source: /kb/a.txt]\n[source: /kb/b.txt]",
        "q2": "[source: /kb/a.txt]\n[source: /kb/c.txt]",
    }
    report = rag_eval.score_cases(cases, lambda query: outputs[query])
    assert report["recall_at_1"] == 0.5
    assert report["recall_at_3"] == 1.0
    assert report["recall_at_8"] == 1.0
    assert report["mrr"] == 0.75
