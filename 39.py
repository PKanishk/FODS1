import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Data Preprocessing
data = pd.read_csv('transaction_data.csv')  # Load your transaction data
# Handle missing values and outliers if needed
# Normalize or scale the features
scaler = StandardScaler()
data[['total_amount', 'num_items']] = scaler.fit_transform(data[['total_amount', 'num_items']])

# Step 2: Feature Selection
X = data[['total_amount', 'num_items']]

# Step 3: Choose the Number of Clusters (K)
# Use the elbow method or silhouette analysis to determine K

# Step 4: Train the K-Means Model
k = 4  # Choose an appropriate value of K
kmeans = KMeans(n_clusters=k, random_state=0)
data['cluster'] = kmeans.fit_predict(X)

# Step 5: Visualize the Clusters
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segmentation Using K-Means')
plt.show()
