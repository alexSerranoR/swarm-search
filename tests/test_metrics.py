"""Metric tests. Expected values are computed by hand (see comments)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.metrics import ndcg_at_k, recall_at_k  # noqa: E402

# ---------------- recall@k ----------------


def test_recall_cuts_at_k():
    # top-2 = [a, b] -> only "a" is relevant -> 1 of 3
    assert recall_at_k(["a", "b", "c", "d"], {"a": 1, "c": 1, "e": 1}, k=2) == pytest.approx(1 / 3)


def test_recall_full_ranking():
    # top-4 contains a and c -> 2 of 3 ("e" never appears)
    assert recall_at_k(["a", "b", "c", "d"], {"a": 1, "c": 1, "e": 1}, k=4) == pytest.approx(2 / 3)


def test_recall_perfect():
    assert recall_at_k(["a", "b"], {"a": 1, "b": 1}, k=10) == pytest.approx(1.0)


def test_recall_no_relevant():
    assert recall_at_k(["a", "b"], {}, k=10) == 0.0


# ---------------- NDCG@k ----------------


def test_ndcg_basic_example():
    # ranked = [a, b, c], relevant: a and c
    # DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5       = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)     = 1 + 0.6309... = 1.6309...
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "b", "c"], {"a": 1, "c": 1}, k=3) == pytest.approx(expected)


def test_ndcg_perfect_is_one():
    assert ndcg_at_k(["a", "b", "x"], {"a": 1, "b": 1}, k=3) == pytest.approx(1.0)


def test_ndcg_relevant_in_second_position():
    # DCG = 1/log2(3), IDCG = 1 -> 0.6309...
    assert ndcg_at_k(["b", "a"], {"a": 1}, k=10) == pytest.approx(1 / math.log2(3))


def test_ndcg_nothing_relevant():
    assert ndcg_at_k(["x", "y"], {"a": 1}, k=10) == 0.0


def test_ndcg_relevant_outside_top_k():
    # "a" is at position 3 but k=2 -> it does not count
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, k=2) == 0.0


def test_ndcg_idcg_is_cut_at_k():
    # 3 relevant docs but k=1: the ideal ranking has 1 slot -> IDCG = 1, DCG = 1
    assert ndcg_at_k(["a"], {"a": 1, "b": 1, "c": 1}, k=1) == pytest.approx(1.0)


def test_ndcg_graded_relevance():
    # grades: a=2, b=1. ranked = [b, a]
    # DCG  = 1/log2(2) + 2/log2(3) = 1 + 1.2619 = 2.2619
    # IDCG = 2/log2(2) + 1/log2(3) = 2 + 0.6309 = 2.6309
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, k=10) == pytest.approx(expected)


def test_ndcg_no_relevant():
    assert ndcg_at_k(["a"], {}, k=10) == 0.0
