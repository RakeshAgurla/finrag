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


def test_regression_gate_flags_a_drop(tmp_path):
    """The gate must actually fail, not just report."""
    import json

    from finrag.eval.metrics import EvalResult, QueryResult
    from finrag.eval.run_eval import check_regression

    baseline = tmp_path / "results.json"
    baseline.write_text(
        json.dumps({"results": [{"config": "hybrid_rrf", "ndcg@5": 0.90, "mrr": 0.80}]})
    )

    degraded = EvalResult(
        config_name="hybrid_rrf",
        per_query=[
            QueryResult("q1", "?", [], set(), {"ndcg@5": 0.50, "mrr": 0.80, "recall@5": 1.0})
        ],
    )
    violations = check_regression([degraded], baseline)
    assert len(violations) == 1 and "ndcg@5" in violations[0]


def test_regression_gate_allows_small_noise(tmp_path):
    import json

    from finrag.eval.metrics import EvalResult, QueryResult
    from finrag.eval.run_eval import check_regression

    baseline = tmp_path / "results.json"
    baseline.write_text(json.dumps({"results": [{"config": "c", "ndcg@5": 0.90}]}))
    within = EvalResult(
        config_name="c",
        per_query=[QueryResult("q1", "?", [], set(), {"ndcg@5": 0.885})],
    )
    assert check_regression([within], baseline) == []


def test_missing_baseline_is_not_a_failure(tmp_path):
    from finrag.eval.run_eval import check_regression

    assert check_regression([], tmp_path / "nope.json") == []
