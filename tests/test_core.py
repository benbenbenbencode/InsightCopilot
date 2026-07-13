import pandas as pd
import pytest

from core.join_engine import get_join_condition, is_allowed_join
from core.anomaly import detect_anomalies
from core.recommender import generate_recommendations


def test_join_engine_known_pair():
    assert get_join_condition("fct_transactions", "dim_users") == (
        "fct_transactions.user_id = dim_users.user_id"
    )


def test_join_engine_reverse_pair():
    assert get_join_condition("dim_users", "fct_transactions") is not None


def test_join_engine_unknown_pair():
    assert get_join_condition("fct_transactions", "some_random_table") is None


def test_is_allowed_join():
    assert is_allowed_join("fct_transactions", "dim_products") is True
    assert is_allowed_join("dim_users", "fct_promotions") is False


def test_anomaly_requires_correct_columns():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(ValueError):
        detect_anomalies(df, value_col="sales", date_col="date")


def test_anomaly_detects_a_spike():
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        "sales": [100, 105, 98, 5000, 102],
    })
    anomalies = detect_anomalies(df, value_col="sales", date_col="date", method="mad")
    assert len(anomalies) >= 1
    assert "2026-01-04" in anomalies["date"].values


def test_recommender_ranks_higher_volume_first():
    df = pd.DataFrame({
        "product": ["A", "A", "B", "B", "B"],
        "sales": [1000, 1000, 10, 10, 10],
    })
    recs = generate_recommendations(df, dimension_col="product", value_col="sales", top_n=2)
    assert recs.iloc[0]["product"] == "A"
