# Customer Intelligence & Recommendation System

## 1. Project Overview
This project applies Unsupervised Learning techniques to analyze customer transaction data, segment users based on their behavior, and provide personalized product recommendations.

## 2. Approach (Methodology)
- **Preprocessing:** Cleaned the 'Online Retail' dataset and engineered RFM (Recency, Frequency, Monetary) features. Applied `StandardScaler` to normalize data.
- **Clustering:** Used K-Means algorithm to group customers. The optimal clusters were determined using the Elbow Method.
- **Anomaly Detection:** Implemented Gaussian Density Estimation to flag unusual purchasing patterns (potential wholesale buyers or outliers).
- **Dimensionality Reduction:** Used PCA (Principal Component Analysis) to visualize high-dimensional RFM data in a 2D space.
- **Recommendations:** Built a Collaborative Filtering system to suggest top products based on cluster-level trends.

## 3. Key Insights & Observations
- **Segments:** Identified distinct groups such as 'Champions' (high spenders) and 'At-Risk' customers.
- **Visualization:** PCA plots confirmed that clusters are well-separated, proving the model's effectiveness.
- **Recommendations:** By using the `Description` column, we provided meaningful product names instead of just IDs, enhancing business utility.

## 4. How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
