"""RRF tests. Expected values computed by hand with k=60."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrievers.rrf import rrf  # noqa: E402


def test_rrf_rewards_consensus():
    # A = [x, z, y], B = [y, x]
    # x: 1/61 + 1/62 = 0.03252   y: 1/63 + 1/61 = 0.03227   z: 1/62 = 0.01613
    assert rrf([["x", "z", "y"], ["y", "x"]]) == ["x", "y", "z"]


def test_rrf_single_list_keeps_order():
    assert rrf([["a", "b", "c"]]) == ["a", "b", "c"]


def test_rrf_includes_docs_from_all_lists():
    assert set(rrf([["a"], ["b"], ["c", "a"]])) == {"a", "b", "c"}


def test_rrf_doc_in_most_lists_wins():
    # "c" is last in every list but appears in all 3
    result = rrf([["a", "c"], ["b", "c"], ["d", "c"]])
    assert result[0] == "c"


def test_rrf_small_k_favors_the_top():
    # A=[x], B=[p, q, y], C=[r, s, y]
    # "x" is 1st in one list; "y" is 3rd in two lists.
    # k=0 : x = 1/1  = 1.0     ; y = 1/3 + 1/3   = 0.667   -> x wins (small k rewards the top)
    # k=60: x = 1/61 = 0.01639 ; y = 1/63 + 1/63 = 0.03175 -> y wins (large k rewards consensus)
    rankings = [["x"], ["p", "q", "y"], ["r", "s", "y"]]
    assert rrf(rankings, k=0).index("x") < rrf(rankings, k=0).index("y")
    assert rrf(rankings, k=60).index("y") < rrf(rankings, k=60).index("x")
