import streamlit as st
import pandas as pd
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load preprocessors and models
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans_model.pkl")

# Optional: reload Agglomerative
agg_model = joblib.load("agg_model.pkl")

st.title("Clustering with KMeans and Agglomerative Models")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    # Preprocess: impute, scale
    X = imputer.transform(df)
    X_scaled = scaler.transform(X)

    # --- KMeans predictions ---
    st.subheader("KMeans Results")
    kmeans_labels = kmeans.predict(X_scaled)
    df["Cluster_KMeans"] = kmeans_labels
    st.write(df[["Cluster_KMeans"]].head())

    # --- Agglomerative Clustering ---
    st.subheader("Agglomerative Results (re-fit)")
    # ⚠ Agglomerative does not support .predict(), so we re-fit here
    agg = AgglomerativeClustering(n_clusters=len(set(kmeans_labels)))
    agg_labels = agg.fit_predict(X_scaled)
    df["Cluster_Agglomerative"] = agg_labels
    st.write(df[["Cluster_Agglomerative"]].head())

    # --- Compare Silhouette Scores ---
    st.subheader("Silhouette Scores")
    kmeans_score = silhouette_score(X_scaled, kmeans_labels)
    agg_score = silhouette_score(X_scaled, agg_labels)

    st.write(f"KMeans Silhouette Score: **{kmeans_score:.4f}**")
    st.write(f"Agglomerative Silhouette Score: **{agg_score:.4f}**")

    # --- Download clustered dataset ---
    st.subheader("Download Results")
    st.download_button(
        label="Download Clustered CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="clustered_data.csv",
        mime="text/csv",
    )
