import pandas as pd


def generate_recommendations(
    df: pd.DataFrame,
    dimension_col: str = "product",
    value_col: str = "sales",
    top_n: int = 5,
):
    """Ranks `dimension_col` (e.g. product) by a composite score.

    Why percentile rank instead of fixed weights (0.6/0.3/0.1 like before):
    total_sales, avg_sales, and frequency are on completely different
    scales, so any fixed weight on the raw values is really just letting
    whichever metric has the largest numbers dominate. Ranking each metric
    to a 0-1 percentile first makes the weights mean what they say they
    mean. The weights below (0.5 volume, 0.3 consistency, 0.2 frequency)
    reflect a simple business judgment call -- total volume matters most,
    but a product that's both high-volume AND consistently high-value AND
    frequently bought is a stronger bet than one that's high-volume from a
    single outlier order. Treat these as a documented starting point, not
    a law -- swap them if your business cares about a different mix.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    agg = (
        df.groupby(dimension_col)
        .agg(
            total_value=(value_col, "sum"),
            avg_value=(value_col, "mean"),
            frequency=(value_col, "count"),
        )
        .reset_index()
    )

    if len(agg) == 0:
        return agg

    agg["volume_rank"] = agg["total_value"].rank(pct=True)
    agg["consistency_rank"] = agg["avg_value"].rank(pct=True)
    agg["frequency_rank"] = agg["frequency"].rank(pct=True)

    WEIGHTS = {"volume_rank": 0.5, "consistency_rank": 0.3, "frequency_rank": 0.2}

    agg["score"] = sum(agg[col] * w for col, w in WEIGHTS.items())

    agg["why"] = agg.apply(
        lambda r: (
            f"volume percentile {r['volume_rank']:.0%}, "
            f"consistency percentile {r['consistency_rank']:.0%}, "
            f"frequency percentile {r['frequency_rank']:.0%}"
        ),
        axis=1,
    )

    return agg.sort_values("score", ascending=False).head(top_n)
