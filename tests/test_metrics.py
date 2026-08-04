import math

from finrag.eval.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_partial():
    assert recall_at_k(["a", "b"], {"a", "c"}, 5) == 0.5


def test_recall_respects_k():
    assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_hit_rate_binary():
    assert hit_rate_at_k(["x", "a"], {"a"}, 5) == 1.0
    assert hit_rate_at_k(["x", "y"], {"a"}, 5) == 0.0


def test_reciprocal_rank_position():
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_ordering_is_one():
    assert math.isclose(ndcg_at_k(["a", "b"], {"a": 2.0, "b": 1.0}, 5), 1.0)


def test_ndcg_penalises_inversion():
    good = ndcg_at_k(["a", "b"], {"a": 2.0, "b": 1.0}, 5)
    bad = ndcg_at_k(["b", "a"], {"a": 2.0, "b": 1.0}, 5)
    assert bad < good


def test_ndcg_empty_relevance():
    assert ndcg_at_k(["a"], {}, 5) == 0.0
