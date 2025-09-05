import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# --- Load models ---
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🧠 Customer Segmentation Predictor")
st.markdown("Manual input (single customer) or CSV upload (batch clustering).")

# --- Encode function ---
def encode_categories(df):
    df = df.copy()
    df['Gender'] = df['Gender'].astype('category').cat.codes.replace(-1, np.nan)
    df['MembershipLevel'] = df['MembershipLevel'].astype('category').cat.codes.replace(-1, np.nan)
    df['Country'] = df['Country'].astype('category').cat.codes.replace(-1, np.nan)
    return df

# --- Tabs ---
tab1, tab2 = st.tabs(["Single Customer Input", "CSV Upload"])

# --------------------- #
# --- Single Customer --- #
# --------------------- #
with tab1:
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    income = st.number_input("Annual Income", 1000.0, 200000.0, 60000.0)
    score = st.slider("Spending Score", 1, 100, 50)
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

    encoded = encode_categories(input_df)
    imputed = pd.DataFrame(imputer.transform(encoded), columns=encoded.columns)
    scaled = pd.DataFrame(scaler.transform(imputed), columns=encoded.columns)

    if st.button("Predict Cluster (Single Customer)"):
        cluster = kmeans.predict(scaled)[0]
        st.success(f"✅ Cluster {cluster} (KMeans)")

# --------------------- #
# --- CSV Upload --- #
# --------------------- #
with tab2:
    st.subheader("Dataset Requirements")
    st.markdown("""
    CSV must have columns: Age, Gender, AnnualIncome, SpendingScore, MembershipLevel, Country
    """)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        # Validate columns
        required_cols = ['Age', 'Gender', 'AnnualIncome', 'SpendingScore', 'MembershipLevel', 'Country']
        missing = [c for c in required_cols if c not in df.columns]
        extra = [c for c in df.columns if c not in required_cols]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        if extra:
            st.warning(f"Extra columns ignored: {extra}")

        df = df[required_cols]
        encoded = encode_categories(df)

        try:
            imputed = pd.DataFrame(imputer.transform(encoded), columns=encoded.columns)
            scaled = pd.DataFrame(scaler.transform(imputed), columns=encoded.columns)
        except ValueError as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        model_choice = st.selectbox("Select Clustering Model", ["KMeans", "Agglomerative"])
        n_clusters = st.number_input("Number of Clusters (Agglomerative)", 2, 20, 3)

        if st.button("Predict Clusters (CSV)"):
            if model_choice == "KMeans":
                df["Cluster_KMeans"] = kmeans.predict(scaled)
                score = silhouette_score(scaled, df["Cluster_KMeans"])
                st.write(df.head())
                st.success(f"KMeans Silhouette Score: {score:.4f}")
            elif model_choice == "Agglomerative":
                if len(scaled) < 2:
                    st.error("Agglomerative requires at least 2 rows.")
                    st.stop()
                agg = AgglomerativeClustering(n_clusters=n_clusters)
                df["Cluster_Agglomerative"] = agg.fit_predict(scaled)
                score = silhouette_score(scaled, df["Cluster_Agglomerative"])
                st.write(df.head())
                st.success(f"Agglomerative Silhouette Score: {score:.4f}")

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Clustered CSV", csv, "clustered_data.csv")
