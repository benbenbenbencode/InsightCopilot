import pandas as pd

Z_SCORE_THRESHOLD = 2.0
MAD_THRESHOLD = 3.5  # standard rule-of-thumb cutoff for modified z-score


def _zscore_anomalies(grouped: pd.DataFrame, value_col: str, threshold: float):
    mean = grouped[value_col].mean()
    std = grouped[value_col].std()

    if std == 0 or pd.isna(std):
        return pd.DataFrame()

    grouped = grouped.copy()
    grouped["z_score"] = (grouped[value_col] - mean) / std
    return grouped[grouped["z_score"].abs() > threshold]


def _mad_anomalies(grouped: pd.DataFrame, value_col: str, threshold: float):
    """Median Absolute Deviation based detection. More robust than z-score
    because it doesn't let one extreme outlier drag the mean/std around --
    a single huge spike day won't mask other genuine anomalies the way it
    can with plain z-score.
    """
    median = grouped[value_col].median()
    abs_dev = (grouped[value_col] - median).abs()
    mad = abs_dev.median()

    if mad == 0:
        return pd.DataFrame()

    grouped = grouped.copy()
    # 0.6745 scales MAD to be comparable to a standard deviation under
    # normality -- this is the standard "modified z-score" formula.
    grouped["modified_z_score"] = 0.6745 * (grouped[value_col] - median) / mad
    return grouped[grouped["modified_z_score"].abs() > threshold]


def detect_anomalies(
    df: pd.DataFrame,
    value_col: str = "sales",
    date_col: str = "date",
    method: str = "mad",
    threshold: float = None,
):
    """Detects anomalous days in a time series.

    method: "zscore" (legacy, sensitive to extreme outliers) or "mad"
    (default, more robust -- recommended unless you have a specific reason
    to match the old behavior).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if value_col not in df.columns or date_col not in df.columns:
        # Fail loudly and specifically instead of throwing a bare KeyError
        # three calls deep, which used to get silently swallowed by a
        # generic try/except in api/main.py.
        raise ValueError(
            f"detect_anomalies expected columns '{value_col}' and "
            f"'{date_col}', got {list(df.columns)}"
        )

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    grouped = df.groupby(date_col)[value_col].sum().reset_index()

    if method == "zscore":
        result = _zscore_anomalies(grouped, value_col, threshold or Z_SCORE_THRESHOLD)
    else:
        result = _mad_anomalies(grouped, value_col, threshold or MAD_THRESHOLD)

    if result.empty:
        return result

    result = result.copy()
    score_col = "z_score" if "z_score" in result.columns else "modified_z_score"
    result["severity"] = pd.cut(
        result[score_col].abs(),
        bins=[0, 3, 5, float("inf")],
        labels=["moderate", "high", "severe"],
    )

    return result
