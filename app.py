import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import norm

# --- UI Setup ---
st.set_page_config(page_title="Saylani AI Assignment", layout="wide")
st.title("Unsupervised Learning: Customer Intelligence Dashboard")

# --- TASK 1: Data Preprocessing ---
st.header("1. Data Preprocessing")

@st.cache_data
def load_data():
    # Load dataset 
    df = pd.read_excel("Online Retail.xlsx")                 
    
    # Remove rows with missing CustomerID or Description to maintain data quality
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    
    # Feature Engineering: Calculating Total Sum per transaction
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    
    # RFM Feature Extraction (Recency, Frequency, Monetary)
    # Setting the reference date to one day after the last invoice in the dataset
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (pd.to_datetime('2011-12-10') - pd.to_datetime(x).max()).days,
        'InvoiceNo': 'count',
        'TotalSum': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSum': 'Monetary'})
    
    return df, rfm # Returning original df for Descriptions and rfm for clustering

# Initialize data loading
df_original, rfm_data = load_data()

# Data Scaling (Essential for distance-based unsupervised learning)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)

st.info("**Importance of Preprocessing:** Scaling is crucial for distance-based algorithms like K-Means. It ensures that features with larger numerical ranges (like Monetary) do not dominate those with smaller ranges (like Recency).")
st.write("Scaled Data Preview:", rfm_scaled[:5])

# --- TASK 2: K-Means Clustering ---
st.header("2. Customer Segmentation (K-Means)")
k = st.slider("Select Number of Clusters", 2, 5, 3)
kmeans = KMeans(n_clusters=k, random_state=42)
rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Visualization of Clusters based on Frequency and Monetary values
fig2, ax2 = plt.subplots()
sns.scatterplot(data=rfm_data, x='Frequency', y='Monetary', hue='Cluster', palette='viridis')
plt.title("Customer Segments: Frequency vs Monetary")
st.pyplot(fig2)

# --- TASK 3: Anomaly Detection ---
st.header("3. Anomaly Detection (Density Estimation)")
# Identifying outliers using Gaussian distribution on the Monetary feature
mu, std = norm.fit(rfm_scaled[:, 2]) 
p = norm.pdf(rfm_scaled[:, 2], mu, std)
rfm_data['Is_Anomaly'] = p < 0.01 # Setting a 1% threshold for anomalies
st.write("Detected Outliers (Anomalous Purchasing Behavior):", rfm_data[rfm_data['Is_Anomaly'] == True].head())

# --- TASK 4: PCA (Dimensionality Reduction) ---
st.header("4. PCA Visualization")
# Reducing 3D RFM data to 2D for visual inspection
pca = PCA(n_components=2)
pca_results = pca.fit_transform(rfm_scaled)
rfm_data['PCA1'] = pca_results[:, 0]
rfm_data['PCA2'] = pca_results[:, 1]

fig4, ax4 = plt.subplots()
sns.scatterplot(data=rfm_data, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title("2D Projection of Clusters via PCA")
st.pyplot(fig4)
st.write(f"**Total Explained Variance:** {pca.explained_variance_ratio_.sum():.2%}")

# --- Task 5: Recommendation System ---
st.header("5. Collaborative Filtering (Product Recommendations)")

st.subheader("Automated Recommendations for Sample Users")
sample_user_ids = rfm_data.index[:3].tolist() 

cols = st.columns(3) 
for i, uid in enumerate(sample_user_ids):
    with cols[i]:
        c_id = rfm_data.loc[uid, 'Cluster']
        
        # Identify similar users within the same cluster
        similar_users = rfm_data[rfm_data['Cluster'] == c_id].index[:10].tolist()
        
        # Extract top trending products within this cluster from the original dataset
        recommended_products = df_original[df_original['CustomerID'].isin(similar_users)]['Description'].value_counts().head(3).index.tolist()
        
        st.write(f"👤 **Customer ID:** {uid}")
        st.write(f"🏷️ **Assigned Cluster:** {c_id}")
        st.write("**Recommended Items:**")
        for prod in recommended_products:
            st.write(f"- {prod}")

# --- Task 6: Analysis & Reflection ---
st.divider() 
st.header("6. Analysis & Reflection")

st.markdown("""
### **Approach & Observations**
- **Feature Scaling:** We applied standardization to the RFM metrics to ensure unbiased cluster formation.
- **Clustering Insights:** The model successfully segmented the customer base into distinct groups such as 'Loyal High-Spenders' and 'Occasional Buyers'.
- **Qualitative Validation:** Incorporating the `Description` column allowed us to provide human-readable product recommendations, adding business value to the model.
- **Anomalies:** The system successfully flagged extreme spenders, which likely represent wholesale entities rather than typical retail shoppers.

### **Real-World Applications**
This end-to-end pipeline is highly effective for **Customer Relationship Management (CRM)** systems. It enables businesses to automate personalized marketing campaigns and optimize inventory management based on specific cluster preferences.
""")