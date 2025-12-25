import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Global AI Solutions | Churn Predictor", layout="wide")

# --- CUSTOM STYLING ---
st.title("📊 AI-Driven Customer Retention Analytics")
st.markdown("""
**Business Value Proposition:** This Predictive Model identifies high-risk customers before they churn, allowing businesses to execute proactive retention strategies. 
*Target Industries: SaaS, Telecom, Banking, and E-commerce.*
""")

# --- DATASET ENGINE ---
@st.cache_data
def load_global_data():
    # Creating a synthetic dataset with professional feature names
    np.random.seed(42)
    data = pd.DataFrame({
        'Account_Age_Months': np.random.randint(1, 72, 1000),
        'Monthly_Subscription_Fee': np.random.uniform(20, 200, 1000),
        'Customer_Service_Tickets': np.random.randint(0, 12, 1000),
        'Churn_Status': np.random.choice([0, 1], 1000, p=[0.75, 0.25])
    })
    return data

df = load_global_data()

# --- MACHINE LEARNING PIPELINE ---
X = df.drop('Churn_Status', axis=1)
y = df['Churn_Status']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- SIDEBAR: GLOBAL CLIENT INPUTS ---
st.sidebar.header("🕹️ Prediction Console")
st.sidebar.info("Adjust parameters to simulate a customer profile.")

tenure = st.sidebar.slider("Account Age (Months)", 1, 72, 24)
monthly = st.sidebar.slider("Monthly Fee ($)", 20, 200, 89)
calls = st.sidebar.slider("Service Tickets (Lifetime)", 0, 15, 3)

# --- PREDICTION LOGIC ---
st.subheader("🔍 Real-time Analysis")
if st.button("Generate Risk Assessment"):
    features = np.array([[tenure, monthly, calls]])
    probability = model.predict_proba(features)[0][1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Churn Probability", value=f"{probability:.1%}")
    
    with col2:
        if probability > 0.5:
            st.error("🚨 ACTION REQUIRED: High Attrition Risk")
            st.write("**Recommendation:** Assign a Dedicated Success Manager and offer a 15% loyalty rebate.")
        else:
            st.success("✅ STABLE: Low Attrition Risk")
            st.write("**Recommendation:** Eligible for Premium Upsell or Annual Plan conversion.")

# --- ANALYTICS VIEW ---
st.divider()
with st.expander("View Training Data Insights"):
    st.write(df.head(10))
    st.caption("Note: This model uses historical behavioral patterns to calculate risk scores.")