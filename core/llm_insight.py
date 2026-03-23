import pandas as pd


def generate_insight(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No data available for analysis."

    df = df.copy()

    # ensure numeric
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)

    # aggregate
    sales_by_product = df.groupby("product")["sales"].sum()

    top_product = sales_by_product.idxmax()
    top_value = sales_by_product.max()

    total_sales = df["sales"].sum()
    avg_sales = df["sales"].mean()

    # simple business logic
    insight = f"""
    📊 **Revenue Analysis**

    - **{top_product}** is the top-performing product with total sales of **{int(top_value)}**, 
      contributing significantly to overall revenue.

    - Total revenue across all products is **{int(total_sales)}**, with an average transaction value of **{round(avg_sales, 2)}**.

    📈 **Business Insight**

    The strong performance of **{top_product}** suggests high customer demand and product-market fit. 
    Consider increasing inventory, prioritizing promotions, or bundling this product to maximize revenue.

    Lower-performing products may require pricing adjustments, marketing boosts, or discontinuation analysis.
    """

    return insight.strip()