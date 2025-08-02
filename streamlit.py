import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Retail Analytics App", layout="wide")

# --- Data load from MySQL ---
@st.cache_data(ttl=600)
def load_data():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Naman@0305',    
            database='retaildb'
        )
        df = pd.read_sql("SELECT * FROM online_retail", conn)
        conn.close()
        # Basic Data Cleaning
        df = df.dropna(subset=['CustomerID'])
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].reset_index(drop=True)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        return df
    except mysql.connector.Error as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_data()

# --- Sidebar for navigation ---
st.sidebar.title("Retail Analytics App")
page = st.sidebar.radio("Navigation", ("Customer Segmentation", "Product Recommendation"))

# --- Customer Segmentation Page ---
if page == "Customer Segmentation":
    st.header("Customer Segmentation (Clustering Module)")
    st.write("""
        Enter Recency, Frequency, and Monetary values to predict customer segment:
        - High-Value
        - Regular
        - Occasional
        - At-Risk
    """)
    col1, col2, col3 = st.columns(3)
    recency = col1.number_input("Recency (days since last purchase)", min_value=0, step=1, value=30)
    frequency = col2.number_input("Frequency (# of purchases)", min_value=0, step=1, value=5)
    monetary = col3.number_input("Monetary (total spend)", min_value=0.0, step=1.0, value=100.0, format="%.2f")

    # --- Compute clusters and assign labels ---
    if st.button("Predict Segment"):
        if df.empty:
            st.error("Data not loaded, cannot perform segmentation.")
        else:
            reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (reference_date - x.max()).days,
                'InvoiceNo': 'nunique',
                'TotalPrice': 'sum'
            }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
            kmeans.fit(rfm_scaled)
            rfm['Cluster'] = kmeans.labels_

            # Assign human-friendly labels
            cluster_avg = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
            medians = cluster_avg.median()
            cluster_labels = {}
            for c in cluster_avg.index:
                vals = cluster_avg.loc[c]
                if (vals['Recency'] <= medians['Recency']) and (vals['Frequency'] >= medians['Frequency']):
                    cluster_labels[c] = 'High-Value'
                elif vals['Frequency'] >= medians['Frequency']:
                    cluster_labels[c] = 'Regular'
                elif vals['Recency'] > medians['Recency']:
                    cluster_labels[c] = 'Occasional'
                else:
                    cluster_labels[c] = 'At-Risk'
            # Predict and show cluster
            input_scaled = scaler.transform([[recency, frequency, monetary]])
            pred_cluster = kmeans.predict(input_scaled)[0]
            pred_label = cluster_labels.get(pred_cluster, "Unknown")
            st.success(f"Predicted Customer Segment: **{pred_label}**")

    st.markdown("---")
    st.subheader("Cluster Analysis Visualizations")

    # --- Cluster Profiles Bar Plot ---
    st.write("**Average RFM Values per Cluster**")
    if not df.empty:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(rfm_scaled)
        rfm['Cluster'] = clusters

        # Bar plot like your example
        cluster_avg = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
        cluster_avg_melted = pd.melt(cluster_avg, id_vars=['Cluster'], value_vars=['Recency', 'Frequency', 'Monetary'])

        plt.figure(figsize=(8,5))
        sns.barplot(x="variable", y="value", hue="Cluster", data=cluster_avg_melted, palette="pastel")
        plt.xlabel("Metric", fontsize=12)
        plt.ylabel("Average Value", fontsize=12)
        plt.title("Customer Cluster Profiles", fontsize=14)
        st.pyplot(plt.gcf())
        plt.clf()

        # --- RFM Scatter Plot for Clusters ---
        st.write("**Clusters by Recency vs Monetary**")
        fig, ax = plt.subplots(figsize=(8,5))
        scatter = ax.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], cmap='Set2', alpha=0.6)
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
        ax.add_artist(legend1)
        ax.set_xlabel("Recency (days)")
        ax.set_ylabel("Monetary (total spend)")
        ax.set_title("Cluster Distribution: Recency vs Monetary")
        st.pyplot(fig)

# --- Product Recommendation Page ---
elif page == "Product Recommendation":
    st.header("Product Recommendation Module")
    st.write("Enter a product name to get 5 similar product recommendations based on purchase patterns.")
    product_name = st.text_input("Product Name")
    if st.button("Get Recommendations"):
        if df.empty:
            st.error("Data not loaded, cannot provide recommendations.")
        else:
            pivot = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)
            if product_name not in pivot.columns:
                st.error("Product not found. Please check spelling or try another product.")
            else:
                similarity_matrix = cosine_similarity(pivot.T)
                sim_df = pd.DataFrame(similarity_matrix, index=pivot.columns, columns=pivot.columns)
                recommendations = sim_df[product_name].sort_values(ascending=False)\
                                    .drop(product_name).head(5)
                st.subheader("Top 5 Similar Products:")
                for i, prod in enumerate(recommendations.index, 1):
                    st.markdown(f"""
    <div style='padding:12px; background-color:#f8fdff; border-radius:8px; margin-bottom:8px;'>
        <span style='color:#12355b; font-weight:bold; font-size:1.1em;'>{i}. {prod}</span>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")
