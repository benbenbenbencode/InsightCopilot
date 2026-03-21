import streamlit as st
import pandas as pd
import altair as alt


def render_chart(df: pd.DataFrame):
    if df is None or df.empty:
        return

    df = df.copy()

    # =========================
    # 🔍 Detect columns
    # =========================
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        st.warning("No numeric data to visualize")
        return

    x_col = cat_cols[0] if cat_cols else df.columns[0]
    y_cols = numeric_cols

    # =========================
    # 🧹 Clean data
    # =========================
    for col in y_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    if df.empty:
        st.warning("No valid data after cleaning")
        return

    df = df.sort_values(y_cols[0], ascending=False).head(10)

    st.subheader("📊 Visualization")

    # =========================
    # 📈 TIME SERIES
    # =========================
    if any(k in x_col.lower() for k in ["date", "time"]):
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
        df = df.sort_values(x_col)
        st.line_chart(df.set_index(x_col)[y_cols])

    # =========================
    # 📊 BAR CHART
    # =========================
    else:
        chart_df = df[[x_col] + y_cols]

        melted = chart_df.melt(
            id_vars=x_col,
            var_name="metric",
            value_name="value"
        )

        chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X(f"{x_col}:N"),
            y="value:Q",
            color="metric:N",
            tooltip=[x_col, "metric", "value"]
        )

        st.altair_chart(chart, use_container_width=True)

    # =========================
    # 🧠 INSIGHTS
    # =========================
    st.markdown("---")
    st.subheader("📊 Key Insights")

    insights = []

    top_row = df.iloc[0]
    bottom_row = df.iloc[-1]

    top_label = top_row[x_col]
    bottom_label = bottom_row[x_col]

    # =========================
    # 🔥 EFFICIENCY
    # =========================
    if len(y_cols) >= 2:
        df["__ratio"] = df[y_cols[0]] / df[y_cols[1]].replace(0, 1)

        best = df.sort_values("__ratio", ascending=False).iloc[0]
        worst = df.sort_values("__ratio", ascending=True).iloc[0]

        insights.append(f"💡 Best efficiency: {best[x_col]}")
        insights.append(f"⚠️ Worst efficiency: {worst[x_col]}")

    # =========================
    # 🔥 BASE
    # =========================
    summary = ", ".join([f"{col}: {top_row[col]:,.0f}" for col in y_cols])
    insights.append(f"🏆 Top {x_col}: {top_label} ({summary})")
    insights.append(f"📉 Lowest {x_col}: {bottom_label}")

    if len(df) > 1:
        second = df.iloc[1]
        diff = ", ".join([
            f"{col}: +{top_row[col] - second[col]:,.0f}" for col in y_cols
        ])
        insights.append(f"🔥 Gap vs 2nd → {diff}")

    # =========================
    # 🚨 ANOMALY DETECTION
    # =========================
    for col in y_cols:
        mean = df[col].mean()
        std = df[col].std()

        if std == 0:
            continue

        high = mean + 2 * std
        low = mean - 2 * std

        for _, row in df.iterrows():
            if row[col] > high:
                insights.append(f"🚨 {row[x_col]} unusually HIGH {col}")
            elif row[col] < low:
                insights.append(f"⚠️ {row[x_col]} unusually LOW {col}")

    # =========================
    # 🔍 ROOT CAUSE (ALWAYS WORKS)
    # =========================
    root_causes = []

    rev_col = None
    promo_col = None

    for c in df.columns:
        if "revenue" in c:
            rev_col = c
        if "promo" in c:
            promo_col = c

    if rev_col and promo_col:
        df["__roi"] = df[rev_col] / (df[promo_col] + 1)

        worst = df.sort_values("__roi").iloc[0]
        best = df.sort_values("__roi", ascending=False).iloc[0]

        root_causes.append(
            f"🔻 {worst[x_col]} underperforms due to weaker ROI vs others"
        )

        root_causes.append(
            f"🔺 {best[x_col]} performs efficiently with strong promo return"
        )

    if rev_col:
        worst_rev = df.sort_values(rev_col).iloc[0]
        root_causes.append(
            f"📉 {worst_rev[x_col]} has the lowest revenue contribution"
        )

    for rc in list(set(root_causes)):
        insights.append(rc)

    # =========================
    # 🚀 RECOMMENDATIONS
    # =========================
    recommendations = []

    if rev_col and promo_col:
        worst = df.sort_values("__roi").iloc[0]
        best = df.sort_values("__roi", ascending=False).iloc[0]

        recommendations.append(
            f"👉 Reduce or optimize promo spend in {worst[x_col]} (low ROI)"
        )

        recommendations.append(
            f"👉 Increase investment in {best[x_col]} (high efficiency)"
        )

    if rev_col:
        lowest = df.sort_values(rev_col).iloc[0]

        recommendations.append(
            f"👉 Investigate demand issues in {lowest[x_col]} (lowest revenue)"
        )

    # =========================
    # 🧾 RENDER
    # =========================
    for i in insights:
        st.write("•", i)

    st.markdown("---")
    st.subheader("🚀 Recommendations")

    for r in list(set(recommendations)):
        st.write(r)