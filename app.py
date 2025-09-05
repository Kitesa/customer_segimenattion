import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load models and preprocessors ---
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🧠 Customer Segmentation Predictor")
st.markdown("Enter customer details below to predict which segment they belong to.")

# --- User input ---
age = st.number_input("Age", min_value=18, max_value=100, value=35)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
income = st.number_input("Annual Income", min_value=1000.0, max_value=200000.0, value=60000.0)
score = st.slider("Spending Score", min_value=1, max_value=100, value=50)
membership = st.selectbox("Membership Level", ["Basic", "Silver", "Gold", "Platinum"])
country = st.selectbox("Country", ["USA", "Canada", "UK", "Germany", "France", "Australia"])

input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "AnnualIncome": income,
    "SpendingScore": score,
    "MembershipLevel": membership,
    "Country": country
}])

# --- Encode categorical features ---
def encode_categories(df):
    df = df.copy()
    df['Gender'] = df['Gender'].astype('category').cat.codes.replace(-1, np.nan)
    df['MembershipLevel'] = df['MembershipLevel'].astype('category').cat.codes.replace(-1, np.nan)
    df['Country'] = df['Country'].astype('category').cat.codes.replace(-1, np.nan)
    return df

encoded = encode_categories(input_df)

# --- Preprocess ---
imputed = pd.DataFrame(imputer.transform(encoded), columns=encoded.columns)
scaled = pd.DataFrame(scaler.transform(imputed), columns=encoded.columns)

# --- Model selection ---
model_choice = st.selectbox(
    "Select Clustering Model",
    ["KMeans", "Agglomerative (not available for single input)"]
)

# --- Predict ---
if st.button("Predict Cluster"):
    if model_choice == "KMeans":
        cluster = kmeans.predict(scaled)[0]
        st.success(f"✅ This customer belongs to **Cluster {cluster}** (KMeans)")

    elif model_choice.startswith("Agglomerative"):
        st.warning("⚠ Agglomerative clustering cannot predict a single customer. Please use KMeans.")

