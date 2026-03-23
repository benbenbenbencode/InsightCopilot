import pandas as pd


def analyze_root_cause(df: pd.DataFrame, anomalies: pd.DataFrame):
    if df is None or df.empty or anomalies is None or anomalies.empty:
        return []

    df = df.copy()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)

    results = []

    for _, row in anomalies.iterrows():
        date = row["date"]

        # filter data for anomaly date
        day_df = df[df["date"] == date]

        if day_df.empty:
            continue

        # aggregate by product
        product_sales = (
            day_df.groupby("product")["sales"]
            .sum()
            .reset_index()
            .sort_values("sales", ascending=False)
        )

        top_product = product_sales.iloc[0]

        results.append({
            "date": date,
            "top_product": top_product["product"],
            "sales": float(top_product["sales"]),
            "reason": f"{top_product['product']} drove unusual sales spike"
        })

    return results