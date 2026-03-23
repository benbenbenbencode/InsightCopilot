import pandas as pd


def detect_anomalies(df: pd.DataFrame, threshold: float = 2.0):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ensure types
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)

    # group by date (daily anomaly)
    grouped = df.groupby("date")["sales"].sum().reset_index()

    mean = grouped["sales"].mean()
    std = grouped["sales"].std()

    if std == 0:
        return pd.DataFrame()

    # z-score
    grouped["z_score"] = (grouped["sales"] - mean) / std

    anomalies = grouped[abs(grouped["z_score"]) > threshold]

    return anomalies