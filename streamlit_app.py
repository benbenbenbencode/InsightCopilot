import streamlit as st
import pandas as pd
import requests

from app.chart import render_chart

st.set_page_config(page_title="AI Decision Intelligence System", layout="wide")

st.title("📊 AI Decision Intelligence System")

question = st.text_input("Ask your data")

if st.button("Run"):

    # 🚀 CALL API
    try:
        response = requests.post(
            "http://127.0.0.1:8000/analyze",
            json={"question": question}
        )
        result = response.json()

    except Exception as e:
        st.error(f"API Error: {e}")
        st.stop()

    # 🧠 SQL
    st.subheader("🧠 Generated SQL")
    st.code(result.get("sql", ""), language="sql")

    # ❌ ERROR
    if "error" in result:
        st.error("Query failed")
        st.text(result.get("error", "Unknown error"))
        st.stop()

    # ✅ SUCCESS
    if "data" in result:
        df = pd.DataFrame(result["data"])

        st.subheader("📊 Data")
        st.dataframe(df)

        # =========================
        # 📈 AUTO CHART
        # =========================
        render_chart(df)

        # =========================
        # 💡 AI INSIGHT
        # =========================
        st.subheader("💡 Insights")
        st.write(result.get("explanation", "No insight available"))

        # =========================
        # 📊 BASIC STATS (SAFE)
        # =========================
        if not df.empty and "sales" in df.columns:

            total_sales = df["sales"].sum()
            avg_sales = df["sales"].mean()

            st.write(f"💰 Total revenue: **{int(total_sales)}**")
            st.write(f"📊 Average sales: **{round(avg_sales, 2)}**")

            # 🔥 only if product exists
            if "product" in df.columns:
                sales_by_product = df.groupby("product")["sales"].sum()

                top_product = sales_by_product.idxmax()
                top_value = sales_by_product.max()

                st.write(
                    f"📈 Top product is **{top_product}** with total sales of **{int(top_value)}**"
                )

        else:
            st.info("Not enough data for stats")

        # =========================
        # 🚨 ANOMALY DETECTION
        # =========================
        st.subheader("🚨 Anomalies Detected")

        anomalies = result.get("anomalies", [])

        if anomalies:
            anomaly_df = pd.DataFrame(anomalies)
            st.dataframe(anomaly_df)

            for _, row in anomaly_df.iterrows():
                st.write(
                    f"⚠️ Unusual sales on **{row['date']}** "
                    f"(z-score: {round(row['z_score'], 2)})"
                )
        else:
            st.info("No anomalies detected")

        # =========================
        # 🔍 ROOT CAUSE
        # =========================
        st.subheader("🔍 Root Cause Analysis")

        root_causes = result.get("root_cause", [])

        if root_causes:
            rc_df = pd.DataFrame(root_causes)
            st.dataframe(rc_df)

            for _, row in rc_df.iterrows():
                st.write(
                    f"📌 On **{row['date']}**, driven by **{row['top_product']}** "
                    f"(sales: {int(row['sales'])})"
                )
        else:
            st.info("No root cause identified")

        # =========================
        # 🚀 AI RECOMMENDATIONS (FINAL FIX)
        # =========================
        st.subheader("🚀 Recommendations")

        if not df.empty:

            group_col = None

            if "product" in df.columns:
                group_col = "product"
            elif "transaction_city" in df.columns:
                group_col = "transaction_city"
            elif "category" in df.columns:
                group_col = "category"

            if group_col and "sales" in df.columns:

                grouped = (
                    df.groupby(group_col)["sales"]
                    .sum()
                    .reset_index()
                    .sort_values("sales", ascending=False)
                )

                st.dataframe(grouped.head(5))

                for _, row in grouped.head(5).iterrows():
                    st.write(
                        f"📌 **{row[group_col]}** → {int(row['sales'])} sales"
                    )

            else:
                st.info("No suitable columns for recommendation")

        else:
            st.info("No data available")

    else:
        st.warning("No result returned")