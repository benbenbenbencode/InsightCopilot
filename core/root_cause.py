import pandas as pd


def analyze_root_cause(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    value_col: str = "sales",
    date_col: str = "date",
    dimension_col: str = "product",
    top_n: int = 3,
):
    """For each anomalous date, breaks the change down by `dimension_col`
    (e.g. product, city) and ranks how much each dimension contributed to
    the gap between that day and the baseline (the average of all non-
    anomalous days), instead of just naming whichever dimension had the
    single highest raw value that day.

    Also flags whether the spike looks "genuine" (spread across many
    transactions) or "concentrated" (driven by a handful of unusually large
    transactions), since those need different follow-up actions.
    """
    if df is None or df.empty or anomalies is None or anomalies.empty:
        return []

    if any(c not in df.columns for c in (value_col, date_col, dimension_col)):
        raise ValueError(
            f"analyze_root_cause expected columns '{value_col}', '{date_col}', "
            f"'{dimension_col}', got {list(df.columns)}"
        )

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    anomaly_dates = set(anomalies[date_col])
    baseline_df = df[~df[date_col].isin(anomaly_dates)]

    # Baseline: average total-per-dimension across all non-anomalous days.
    baseline_by_dim = (
        baseline_df.groupby(dimension_col)[value_col].sum()
        / max(baseline_df[date_col].nunique(), 1)
    )

    results = []

    for _, row in anomalies.iterrows():
        date = row[date_col]
        day_df = df[df[date_col] == date]

        if day_df.empty:
            continue

        actual_by_dim = day_df.groupby(dimension_col)[value_col].sum()

        contribution = (actual_by_dim - baseline_by_dim.reindex(actual_by_dim.index).fillna(0))
        contribution = contribution.sort_values(ascending=False)

        total_delta = contribution.sum()
        top_contributors = contribution.head(top_n)

        # Genuine-vs-concentrated check: how many transactions made up the
        # top contributor's value that day, vs the baseline transaction
        # count for that same dimension.
        top_dim = top_contributors.index[0] if len(top_contributors) else None
        txn_count_today = (
            day_df[day_df[dimension_col] == top_dim].shape[0] if top_dim else 0
        )
        baseline_txn_count = (
            baseline_df[baseline_df[dimension_col] == top_dim].shape[0]
            / max(baseline_df[date_col].nunique(), 1)
            if top_dim else 0
        )

        concentration = "concentrated (few large transactions)" if (
            top_dim and baseline_txn_count > 0 and txn_count_today < baseline_txn_count * 1.5
        ) else "spread across many transactions"

        results.append({
            "date": date,
            "total_delta_vs_baseline": float(total_delta),
            "top_contributors": [
                {"dimension": dim, "delta_vs_baseline": float(val)}
                for dim, val in top_contributors.items()
            ],
            "pattern": concentration,
            "reason": (
                f"{top_dim} contributed the most to the change "
                f"({top_contributors.iloc[0]:+.0f} vs baseline), pattern: {concentration}"
                if top_dim else "No single dimension stood out."
            ),
        })

    return results
