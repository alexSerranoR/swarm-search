"""Tests de la fase 1. Los valores esperados están calculados a mano (mira los comentarios)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.metrics import ndcg_at_k, recall_at_k  # noqa: E402

# ---------------- recall@k ----------------

def test_recall_corta_en_k():
    # top-2 = [a, b] -> solo "a" es relevante -> 1 de 3
    assert recall_at_k(["a", "b", "c", "d"], {"a": 1, "c": 1, "e": 1}, k=2) == pytest.approx(1 / 3)


def test_recall_ranking_entero():
    # top-4 contiene a y c -> 2 de 3 ("e" nunca aparece)
    assert recall_at_k(["a", "b", "c", "d"], {"a": 1, "c": 1, "e": 1}, k=4) == pytest.approx(2 / 3)


def test_recall_perfecto():
    assert recall_at_k(["a", "b"], {"a": 1, "b": 1}, k=10) == pytest.approx(1.0)


def test_recall_sin_relevantes():
    assert recall_at_k(["a", "b"], {}, k=10) == 0.0


# ---------------- NDCG@k ----------------

def test_ndcg_ejemplo_basico():
    # ranked = [a, b, c], relevantes a y c
    # DCG  = 1/log2(2) + 0 + 1/log2(4)      = 1 + 0.5        = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)          = 1 + 0.6309...  = 1.6309...
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "b", "c"], {"a": 1, "c": 1}, k=3) == pytest.approx(expected)


def test_ndcg_perfecto_es_uno():
    assert ndcg_at_k(["a", "b", "x"], {"a": 1, "b": 1}, k=3) == pytest.approx(1.0)


def test_ndcg_relevante_en_segunda_posicion():
    # DCG = 1/log2(3), IDCG = 1  -> 0.6309...
    assert ndcg_at_k(["b", "a"], {"a": 1}, k=10) == pytest.approx(1 / math.log2(3))


def test_ndcg_nada_relevante():
    assert ndcg_at_k(["x", "y"], {"a": 1}, k=10) == 0.0


def test_ndcg_relevante_fuera_del_top_k():
    # "a" está en la posición 3 pero k=2 -> no cuenta
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, k=2) == 0.0


def test_ndcg_idcg_se_corta_en_k():
    # 3 relevantes pero k=1: el ideal solo tiene 1 posición -> IDCG = 1, DCG = 1
    assert ndcg_at_k(["a"], {"a": 1, "b": 1, "c": 1}, k=1) == pytest.approx(1.0)


def test_ndcg_grados_de_relevancia():
    # grados: a=2, b=1. ranked = [b, a]
    # DCG  = 1/log2(2) + 2/log2(3) = 1 + 1.2619 = 2.2619
    # IDCG = 2/log2(2) + 1/log2(3) = 2 + 0.6309 = 2.6309
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, k=10) == pytest.approx(expected)


def test_ndcg_sin_relevantes():
    assert ndcg_at_k(["a"], {}, k=10) == 0.0
