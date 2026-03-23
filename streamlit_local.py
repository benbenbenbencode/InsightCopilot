import streamlit as st
from core.pipeline import run_pipeline
from app.chart import render_chart
from recommender import generate_recommendations

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("📊 AI Data Analyst (Local)")

question = st.text_input("Ask your data")

if st.button("Run"):
    result = run_pipeline(question)

    st.subheader("🧠 Generated SQL")
    st.code(result.get("sql", ""), language="sql")

    # ❌ ERROR
    if "error" in result:
        st.error("Query failed")
        st.text(result.get("error", "Unknown error"))

    # ✅ SUCCESS
    elif "data" in result:
        df = result["data"]

        st.subheader("📊 Data")
        st.dataframe(df)

        # 🔥 Chart
        render_chart(df)

        # 💡 Insights
        if "explanation" in result:
            st.subheader("💡 Insights")
            st.write(result["explanation"])

        # =========================
        # 🚀 NEW: RECOMMENDATION
        # =========================
        try:
            if all(col in df.columns for col in ["product", "sales", "date"]):

                st.subheader("🚀 AI Recommendations")

                top_products = generate_recommendations(df)

                st.dataframe(top_products)

                # 🔥 Business explanation layer
                for _, row in top_products.iterrows():
                    st.write(
                        f"📌 **{row['product']}** is recommended due to strong performance and positive growth trend."
                    )

            else:
                st.info("Recommendation requires columns: product, sales, date")

        except Exception as e:
            st.warning("Recommendation engine failed")
            st.text(str(e))

    else:
        st.warning("No result returned")