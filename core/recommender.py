import pandas as pd

def generate_recommendations(df: pd.DataFrame, top_n: int = 5):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ensure numeric
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)

    # 🔥 aggregate metrics
    agg = (
        df.groupby("product")
        .agg(
            total_sales=("sales", "sum"),
            avg_sales=("sales", "mean"),
            frequency=("sales", "count")
        )
        .reset_index()
    )

    # 🔥 SCORING MODEL (THIS IS THE KEY)
    agg["score"] = (
        agg["total_sales"] * 0.6 +
        agg["avg_sales"] * 0.3 +
        agg["frequency"] * 0.1
    )

    # normalize (optional but nice)
    if agg["score"].max() != 0:
        agg["score"] = agg["score"] / agg["score"].max()

    # rank
    return agg.sort_values("score", ascending=False).head(top_n)