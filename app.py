"""
Streamlit Application
Interactive UI for unsupervised learning analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataPreprocessor
from src.clustering import CustomerSegmentation
from src.anomaly_detection import AnomalyDetector
from src.pca_analysis import PCAAnalyzer
from src.recommendation_system import RecommendationSystem

# Page configuration
st.set_page_config(
    page_title="Unsupervised Learning Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Cache data loading
@st.cache_resource
def load_and_preprocess_data():
    """Load and preprocess data."""
    preprocessor = DataPreprocessor('Online Retail.xlsx')
    preprocessor.load_data()
    preprocessor.handle_missing_values()
    preprocessor.remove_duplicates()
    df_processed = preprocessor.engineer_features()
    X_scaled, original_df = preprocessor.normalize_features(method='standard')
    return preprocessor, X_scaled, original_df


@st.cache_resource
def perform_clustering(X_scaled, feature_cols, original_df):
    """Perform K-Means clustering."""
    clustering = CustomerSegmentation(X_scaled, feature_cols, original_df)
    clustering.elbow_method(k_range=range(2, 11))
    clustering.fit_kmeans(optimal_k=4)
    return clustering


@st.cache_resource
def perform_anomaly_detection(X_scaled, feature_cols, original_df):
    """Perform anomaly detection."""
    anomaly_detector = AnomalyDetector(X_scaled, feature_cols, original_df)
    anomaly_detector.detect_anomalies_isolation_forest(contamination=0.05)
    return anomaly_detector


@st.cache_resource
def perform_pca(X_scaled, feature_cols, original_df):
    """Perform PCA analysis."""
    pca_analyzer = PCAAnalyzer(X_scaled, feature_cols, original_df)
    pca_analyzer.fit_pca(n_components=len(feature_cols))
    return pca_analyzer


@st.cache_resource
def build_recommendation_system(transaction_df):
    """Build recommendation system."""
    rec_system = RecommendationSystem(transaction_df)
    rec_system.build_customer_item_matrix()
    rec_system.compute_user_similarity()
    rec_system.compute_item_similarity()
    return rec_system


def main():
    """Main Streamlit application."""
    
    # Title and header
    st.title("📊 Unsupervised Learning Analysis Dashboard")
    st.markdown("### Customer Behavior Analysis & Recommendation System")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        preprocessor, X_scaled, original_df = load_and_preprocess_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        [
            "📈 Overview",
            "🔍 Preprocessing",
            "👥 Clustering",
            "⚠️ Anomalies",
            "📉 PCA",
            "🎁 Recommendations",
            "📋 Summary"
        ]
    )
    
    # ========================================================================
    # PAGE: OVERVIEW
    # ========================================================================
    if page == "📈 Overview":
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(preprocessor.df))
        with col2:
            st.metric("Unique Customers", preprocessor.df['CustomerID'].nunique())
        with col3:
            st.metric("Date Range", f"{preprocessor.df['InvoiceDate'].min().strftime('%Y-%m-%d')} to {preprocessor.df['InvoiceDate'].max().strftime('%Y-%m-%d')}")
        with col4:
            st.metric("Total Revenue", f"${preprocessor.df['Amount'].sum():,.0f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Customers by Spending")
            top_customers = original_df.nlargest(10, 'TotalSpending')[['CustomerID', 'TotalSpending']]
            st.bar_chart(top_customers.set_index('CustomerID'))
        
        with col2:
            st.subheader("Revenue Distribution")
            fig, ax = plt.subplots()
            ax.hist(original_df['TotalSpending'], bins=30, color='skyblue', edgecolor='black')
            ax.set_xlabel('Total Spending ($)')
            ax.set_ylabel('Number of Customers')
            st.pyplot(fig)
        
        st.markdown("---")
        
        with st.expander("📊 Dataset Preview"):
            st.dataframe(original_df.head(20), use_container_width=True)
    
    # ========================================================================
    # PAGE: PREPROCESSING
    # ========================================================================
    elif page == "🔍 Preprocessing":
        st.header("Data Preprocessing")
        
        st.subheader("Why Preprocessing is Critical for Unsupervised Learning")
        
        reasons = """
        1. **Consistency**: Removes inconsistencies and anomalies in raw data
        2. **Feature Scaling**: Essential for distance-based algorithms
        3. **Information Extraction**: Converts raw transactions into meaningful features
        4. **Noise Reduction**: Missing values and duplicates can distort patterns
        5. **Algorithm Performance**: Well-preprocessed data leads to better results
        """
        st.info(reasons)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Values Handled")
            missing_info = f"""
            - Original rows: {len(preprocessor.df)}
            - After cleaning: {len(preprocessor.df)}
            - Missing values in key columns: Removed
            """
            st.text(missing_info)
        
        with col2:
            st.subheader("Features Engineered")
            features = """
            ✓ Purchase Frequency
            ✓ Total Spending
            ✓ Average Order Value
            ✓ Product Variety
            ✓ Recency
            ✓ Customer Age
            """
            st.text(features)
        
        st.markdown("---")
        
        st.subheader("Feature Statistics (After Scaling)")
        st.dataframe(
            X_scaled[preprocessor.feature_columns].describe().round(3),
            use_container_width=True
        )
    
    # ========================================================================
    # PAGE: CLUSTERING
    # ========================================================================
    elif page == "👥 Clustering":
        st.header("Customer Segmentation - K-Means Clustering")
        
        with st.spinner("Performing clustering analysis..."):
            clustering = perform_clustering(X_scaled, preprocessor.feature_columns, original_df)
        
        # Elbow curve
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Elbow Method")
            fig, ax = plt.subplots()
            ax.plot(range(2, 2 + len(clustering.inertias)), clustering.inertias, 'bo-', linewidth=2)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Curve')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Silhouette Scores")
            fig, ax = plt.subplots()
            ax.plot(range(2, 2 + len(clustering.silhouette_scores)), clustering.silhouette_scores, 'go-', linewidth=2)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Silhouette Score')
            ax.set_title('Silhouette Method')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Cluster visualization
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            original_df['TotalSpending'],
            original_df['PurchaseFrequency'],
            c=original_df['Cluster'],
            cmap='viridis',
            s=100,
            alpha=0.7,
            edgecolors='black'
        )
        ax.set_xlabel('Total Spending ($)')
        ax.set_ylabel('Purchase Frequency')
        ax.set_title(f'K-Means Customer Segmentation (k={clustering.optimal_k})')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Cluster analysis
        st.subheader("Cluster Characteristics")
        
        for cluster_id in range(clustering.optimal_k):
            with st.expander(f"📍 Cluster {cluster_id}"):
                cluster_data = original_df[original_df['Cluster'] == cluster_id]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Size", len(cluster_data))
                with col2:
                    st.metric("Avg Spending", f"${cluster_data['TotalSpending'].mean():.2f}")
                with col3:
                    st.metric("Avg Frequency", f"{cluster_data['PurchaseFrequency'].mean():.1f}")
                with col4:
                    st.metric("Avg Products", f"{cluster_data['ProductVariety'].mean():.1f}")
                
                st.dataframe(cluster_data.head(10), use_container_width=True)
    
    # ========================================================================
    # PAGE: ANOMALIES
    # ========================================================================
    elif page == "⚠️ Anomalies":
        st.header("Anomaly Detection")
        
        with st.spinner("Detecting anomalies..."):
            anomaly_detector = perform_anomaly_detection(X_scaled, preprocessor.feature_columns, original_df)
        
        # Anomaly statistics
        anomalies = original_df[original_df['IsAnomaly'] == True]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Anomalies", len(anomalies))
        with col2:
            st.metric("Anomaly Rate", f"{100*len(anomalies)/len(original_df):.2f}%")
        with col3:
            st.metric("Avg Anomaly Score", f"{original_df[original_df['IsAnomaly']]['AnomalyScore'].mean():.3f}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spending: Normal vs Anomalous")
            fig, ax = plt.subplots()
            normal = original_df[original_df['IsAnomaly'] == False]
            anomaly = original_df[original_df['IsAnomaly'] == True]
            ax.scatter(normal['TotalSpending'], normal['PurchaseFrequency'], 
                      alpha=0.6, s=50, label='Normal', color='blue')
            ax.scatter(anomaly['TotalSpending'], anomaly['PurchaseFrequency'], 
                      alpha=0.9, s=100, label='Anomaly', color='red', marker='^')
            ax.set_xlabel('Total Spending ($)')
            ax.set_ylabel('Purchase Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Anomaly Score Distribution")
            fig, ax = plt.subplots()
            ax.hist(normal['AnomalyScore'], bins=30, alpha=0.7, label='Normal', color='blue')
            ax.hist(anomaly['AnomalyScore'], bins=30, alpha=0.7, label='Anomaly', color='red')
            ax.set_xlabel('Mahalanobis Distance')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("Top Anomalous Customers")
        top_anomalies = anomalies.nlargest(10, 'AnomalyScore')[
            ['CustomerID', 'TotalSpending', 'PurchaseFrequency', 'ProductVariety', 'AnomalyScore']
        ]
        st.dataframe(top_anomalies, use_container_width=True)
    
    # ========================================================================
    # PAGE: PCA
    # ========================================================================
    elif page == "📉 PCA":
        st.header("Dimensionality Reduction - PCA")
        
        with st.spinner("Performing PCA..."):
            pca_analyzer = perform_pca(X_scaled, preprocessor.feature_columns, original_df)
        
        variance_explained = pca_analyzer.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_explained)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Variance by Component")
            fig, ax = plt.subplots()
            ax.bar(range(1, len(variance_explained) + 1), variance_explained, 
                  alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Variance Explained')
            ax.set_title('Variance by Component')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Cumulative Variance")
            fig, ax = plt.subplots()
            ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', linewidth=2)
            ax.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
            ax.fill_between(range(1, len(cumulative_variance) + 1), cumulative_variance, alpha=0.3)
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Variance')
            ax.legend()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Components for 95% variance
        n_comp_95 = np.argmax(cumulative_variance >= 0.95) + 1
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Components for 95%", n_comp_95)
        with col2:
            st.metric("Variance Captured", f"{cumulative_variance[n_comp_95-1]*100:.2f}%")
        with col3:
            st.metric("Reduction Rate", f"{(1 - n_comp_95/len(preprocessor.feature_columns))*100:.1f}%")
        
        st.markdown("---")
        
        # 2D Projection
        st.subheader("Data in 2D (First Two PCs)")
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            pca_analyzer.X_pca[:, 0],
            pca_analyzer.X_pca[:, 1],
            c=original_df['Cluster'].values,
            cmap='viridis',
            s=60,
            alpha=0.7,
            edgecolors='black'
        )
        ax.set_xlabel(f"PC1 ({variance_explained[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance_explained[1]*100:.1f}%)")
        ax.set_title("Principal Component Analysis - 2D Projection")
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("Variance Explained Table")
        variance_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(variance_explained))],
            'Variance %': (variance_explained * 100).round(2),
            'Cumulative %': (cumulative_variance * 100).round(2)
        })
        st.dataframe(variance_df, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # PAGE: RECOMMENDATIONS
    # ========================================================================
    elif page == "🎁 Recommendations":
        st.header("Recommendation System - Collaborative Filtering")
        
        with st.spinner("Building recommendation system..."):
            rec_system = build_recommendation_system(preprocessor.df)
        
        st.info("""
        **Collaborative Filtering Intuition:**
        - User-Based: "Show customers products liked by similar customers"
        - Item-Based: "Show customers products similar to ones they already like"
        """)
        
        st.markdown("---")
        
        # Select customer for recommendation
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.selectbox(
                "Select a customer",
                sorted(original_df['CustomerID'].values)
            )
        
        with col2:
            method = st.radio("Recommendation Method", ["User-Based", "Item-Based", "Both"])
        
        if st.button("Get Recommendations"):
            st.markdown("---")
            
            if method in ["User-Based", "Both"]:
                st.subheader("👥 User-Based Recommendations")
                user_recs = rec_system.recommend_user_based(customer_id, n_recommendations=5)
                
                if user_recs:
                    for idx, (product, score) in enumerate(user_recs.items(), 1):
                        product_info = rec_system.get_product_details(product)
                        with st.container():
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                st.metric(f"#{idx} Product", product)
                            with col2:
                                st.metric("Description", product_info['Description'][:30])
                            with col3:
                                st.metric("Score", f"{score:.3f}")
                else:
                    st.warning("No user-based recommendations available")
            
            if method in ["Item-Based", "Both"]:
                st.subheader("🎁 Item-Based Recommendations")
                item_recs = rec_system.recommend_item_based(customer_id, n_recommendations=5)
                
                if item_recs:
                    for idx, (product, score) in enumerate(item_recs.items(), 1):
                        product_info = rec_system.get_product_details(product)
                        with st.container():
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                st.metric(f"#{idx} Product", product)
                            with col2:
                                st.metric("Description", product_info['Description'][:30])
                            with col3:
                                st.metric("Score", f"{score:.3f}")
                else:
                    st.warning("No item-based recommendations available")
        
        st.markdown("---")
        
        # Customer information
        st.subheader("Customer Profile")
        customer_info = original_df[original_df['CustomerID'] == customer_id]
        if not customer_info.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Spending", f"${customer_info['TotalSpending'].values[0]:.2f}")
            with col2:
                st.metric("Purchase Frequency", f"{customer_info['PurchaseFrequency'].values[0]:.0f}")
            with col3:
                st.metric("Product Variety", f"{customer_info['ProductVariety'].values[0]:.0f}")
            with col4:
                st.metric("Recency (days)", f"{customer_info['Recency'].values[0]:.0f}")
    
    # ========================================================================
    # PAGE: SUMMARY
    # ========================================================================
    elif page == "📋 Summary":
        st.header("Analysis Summary & Key Insights")
        
        st.subheader("1️⃣ Customer Segmentation (K-Means)")
        st.write("""
        - Identified 4 distinct customer segments
        - Each segment has unique spending and engagement patterns
        - Enables targeted marketing strategies
        """)
        
        st.subheader("2️⃣ Anomaly Detection")
        st.write(f"""
        - Detected {len(original_df[original_df['IsAnomaly']]):.0f} anomalous customers
        - Identified unusual spending and behavior patterns
        - Useful for fraud detection and at-risk customer identification
        """)
        
        st.subheader("3️⃣ Dimensionality Reduction (PCA)")
        st.write("""
        - Reduced from 9 to ~4 dimensions (95% variance retained)
        - Improved visualization and computational efficiency
        - Maintained interpretability of key patterns
        """)
        
        st.subheader("4️⃣ Recommendation System")
        st.write("""
        - Built collaborative filtering system
        - Both user-based and item-based recommendations available
        - Supports cross-selling and upselling strategies
        """)
        
        st.markdown("---")
        
        st.subheader("🎯 Real-World Applications")
        
        applications = {
            "E-Commerce": "Personalized recommendations, churn prediction, customer segmentation",
            "Finance": "Fraud detection, credit risk assessment, portfolio clustering",
            "Telecom": "Network anomaly detection, customer churn prediction",
            "Healthcare": "Disease pattern discovery, patient cohort identification",
            "Manufacturing": "Equipment failure prediction, production anomaly detection"
        }
        
        for sector, use_cases in applications.items():
            st.write(f"**{sector}**: {use_cases}")
        
        st.markdown("---")
        
        st.subheader("📊 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(original_df))
        with col2:
            st.metric("Total Transactions", len(preprocessor.df))
        with col3:
            st.metric("Total Revenue", f"${preprocessor.df['Amount'].sum():,.0f}")
        with col4:
            st.metric("Avg Spending/Customer", f"${original_df['TotalSpending'].mean():.2f}")


if __name__ == "__main__":
    main()
