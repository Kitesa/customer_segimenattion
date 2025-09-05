import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# --------------------- #
# --- Load Models --- #
# --------------------- #
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🧠 Customer Segmentation Predictor")
st.markdown("You can either enter a single customer manually or upload a CSV for batch clustering.")

# --------------------- #
# --- Tabs for input --- #
# --------------------- #
tab1, tab2 = st.tabs(["Single Customer Input", "CSV Upload"])

# --------------------- #
# --- Helper Function --- #
# --------------------- #
def encode_categories(df):
    df = df.copy()
    df['Gender'] = df['Gender'].astype('category').cat.codes.replace(-1, np.nan)
    df['MembershipLevel'] = df['MembershipLevel'].astype('category').cat.codes.replace(-1, np.nan)
    df['Country'] = df['Country'].astype('category').cat.codes.replace(-1, np.nan)
    return df

# --------------------- #
# --- Single Customer --- #
# --------------------- #
with tab1:
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

    if st.button("Predict Cluster (Single Customer)"):
        try:
            encoded = encode_categories(input_df)
            imputed = pd.DataFrame(imputer.transform(encoded), columns=encoded.columns)
            scaled = pd.DataFrame(scaler.transform(imputed), columns=encoded.columns)

            cluster = kmeans.predict(scaled)[0]
            st.success(f"✅ This customer belongs to **Cluster {cluster}** (KMeans)")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --------------------- #
# --- CSV Upload --- #
# --------------------- #
with tab2:
    st.subheader("Dataset Requirements")
    st.markdown("""
    Your CSV must contain the following columns exactly (case-sensitive):
    
    | Column Name         | Type      | Notes                                  |
    |--------------------|-----------|---------------------------------------|
    | Age                 | int       | Customer age, 18+                      |
    | Gender              | categorical | 'Male', 'Female', or 'Other'         |
    | AnnualIncome        | float     | Annual income in dollars               |
    | SpendingScore       | int       | Score between 1 and 100                |
    | MembershipLevel     | categorical | 'Basic', 'Silver', 'Gold', 'Platinum'|
    | Country             | categorical | e.g., 'USA', 'Canada', etc.          |
    """)

    uploaded_file = st.file_uploader("Upload CSV file for batch clustering", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        # --- Check required columns ---
        required_cols = ['Age', 'Gender', 'AnnualIncome', 'SpendingScore', 'MembershipLevel', 'Country']
        missing_cols = [c for c in required_cols if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in required_cols]

        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()
        if extra_cols:
            st.warning(f"⚠ Extra columns will be ignored: {extra_cols}")

        # Keep only required columns
        df = df[required_cols]

        # Encode, impute, and scale
        try:
            encoded = encode_categories(df)
            imputed = pd.DataFrame(imputer.transform(encoded), columns=encoded.columns)
            scaled = pd.DataFrame(scaler.transform(imputed), columns=encoded.columns)
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        # --- Model selection ---
        model_choice = st.selectbox("Select Clustering Model", ["KMeans", "Agglomerative"])
        n_clusters = 3
        if model_choice == "Agglomerative":
            n_clusters = st.number_input("Number of Clusters for Agglomerative", min_value=2, max_value=20, value=3)

        if st.button("Predict Clusters (CSV)"):
            try:
                if model_choice == "KMeans":
                    df["Cluster_KMeans"] = kmeans.predict(scaled)
                    score = silhouette_score(scaled, df["Cluster_KMeans"])
                    st.write(df.head())
                    st.success(f"KMeans Silhouette Score: {score:.4f}")

                elif model_choice == "Agglomerative":
                    agg = AgglomerativeClustering(n_clusters=n_clusters)
                    df["Cluster_Agglomerative"] = agg.fit_predict(scaled)
                    score = silhouette_score(scaled, df["Cluster_Agglomerative"])
                    st.write(df.head())
                    st.success(f"Agglomerative Silhouette Score: {score:.4f}")

                # Download CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Clustered CSV", data=csv, file_name="clustered_data.csv")

            except Exception as e:
                st.error(f"Error during clustering: {e}")
