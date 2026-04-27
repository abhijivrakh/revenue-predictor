
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Revenue Intelligence Predictor",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
}
.title-text {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
}
.subtitle-text {
    font-size: 18px;
    color: #4b5563;
}
.result-box {
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0px 8px 28px rgba(0,0,0,0.15);
}
.recommendation-box {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #2563eb;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

bundle = joblib.load("revenue_prediction_model.pkl")
model = bundle["model"]
metrics = bundle["metrics"]

st.markdown('<div class="title-text">Revenue Intelligence Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Advanced ML-powered revenue prediction using marketing spend and business signals.</div>', unsafe_allow_html=True)

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Business Input Panel")

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        product_category = st.selectbox("Product Category", ["Electronics", "Fashion", "Grocery", "Home Decor", "Beauty", "Sports"])
        campaign_type = st.selectbox("Campaign Type", ["Festival Offer", "Flash Sale", "Brand Awareness", "New Launch", "Clearance Sale", "Regular Campaign"])
        customer_segment = st.selectbox("Customer Segment", ["Budget", "Regular", "Premium", "Enterprise"])

    with col2:
        tv_spend = st.number_input("TV Ad Spend", min_value=0, value=80000, step=5000)
        social_spend = st.number_input("Social Media Spend", min_value=0, value=60000, step=5000)
        google_spend = st.number_input("Google Ads Spend", min_value=0, value=75000, step=5000)
        email_spend = st.number_input("Email Marketing Spend", min_value=0, value=20000, step=2000)

    col3, col4, col5 = st.columns(3)

    with col3:
        influencer_spend = st.number_input("Influencer Marketing Spend", min_value=0, value=40000, step=5000)

    with col4:
        discount = st.slider("Discount Percentage", 0.0, 45.0, 15.0)

    with col5:
        website_visits = st.number_input("Website Visits", min_value=0, value=30000, step=1000)

    previous_revenue = st.number_input("Previous Month Revenue", min_value=0, value=350000, step=10000)

    month = st.selectbox("Month", list(range(1, 13)))
    day = st.selectbox("Day", list(range(1, 29)))
    year = st.selectbox("Year", [2024, 2025, 2026])

    input_data = pd.DataFrame({
        "Region": [region],
        "Product_Category": [product_category],
        "Campaign_Type": [campaign_type],
        "Customer_Segment": [customer_segment],
        "TV_Ad_Spend": [tv_spend],
        "Social_Media_Spend": [social_spend],
        "Google_Ads_Spend": [google_spend],
        "Email_Marketing_Spend": [email_spend],
        "Influencer_Marketing_Spend": [influencer_spend],
        "Discount_Percentage": [discount],
        "Website_Visits": [website_visits],
        "Previous_Month_Revenue": [previous_revenue],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "DayOfWeek": [2]
    })

with right:
    st.subheader("Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", metrics["R2 Score"])
    m2.metric("MAE", f"₹{metrics['MAE']:,.0f}")
    m3.metric("RMSE", f"₹{metrics['RMSE']:,.0f}")

    st.info("Model used: XGBoost Regressor with preprocessing pipeline and hyperparameter tuning.")

    if st.button("Predict Revenue", use_container_width=True):
        prediction = model.predict(input_data)[0]

        if prediction < 800000:
            level = "Low Revenue"
            recommendation = "Revenue is expected to be lower. Improve campaign targeting, increase high-performing digital channels, and review discount strategy."
        elif prediction < 1300000:
            level = "Medium Revenue"
            recommendation = "Revenue is stable. Optimize Google Ads and social media spend while monitoring return on marketing investment."
        else:
            level = "High Revenue"
            recommendation = "Revenue potential is strong. Maintain current campaign strategy and scale the best-performing channels carefully."

        st.markdown(f'''
        <div class="result-box">
            <h2>Predicted Revenue</h2>
            <h1>₹{prediction:,.2f}</h1>
            <h3>{level}</h3>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("### Business Recommendation")
        st.markdown(f'''
        <div class="recommendation-box">
            {recommendation}
        </div>
        ''', unsafe_allow_html=True)

        spends = {
            "TV Ads": tv_spend,
            "Social Media": social_spend,
            "Google Ads": google_spend,
            "Email": email_spend,
            "Influencer": influencer_spend
        }

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(spends.keys()),
            y=list(spends.values())
        ))
        fig.update_layout(
            title="Marketing Spend Distribution",
            xaxis_title="Channel",
            yaxis_title="Spend",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file for bulk revenue prediction", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    if "Date" in batch_df.columns:
        batch_df["Date"] = pd.to_datetime(batch_df["Date"])
        batch_df["Year"] = batch_df["Date"].dt.year
        batch_df["Month"] = batch_df["Date"].dt.month
        batch_df["Day"] = batch_df["Date"].dt.day
        batch_df["DayOfWeek"] = batch_df["Date"].dt.dayofweek
        batch_df = batch_df.drop(columns=["Date"])

    batch_df = batch_df.drop(columns=["Revenue", "Revenue_Level"], errors="ignore")

    predictions = model.predict(batch_df)
    batch_df["Predicted_Revenue"] = predictions

    st.dataframe(batch_df, use_container_width=True)

    csv = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions CSV",
        csv,
        "predicted_revenue_output.csv",
        "text/csv",
        use_container_width=True
    )

st.caption("Built as an end-to-end ML deployment proof using XGBoost, Streamlit, and business-focused revenue intelligence.")
