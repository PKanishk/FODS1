import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and preprocess your dataset
# Replace 'your_data.csv' with the path to your dataset file
data = pd.read_csv('your_data.csv')

# Perform data preprocessing as needed (e.g., handling missing values, encoding categorical variables)

# Select relevant features for clustering or perform feature engineering

# Scale/normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Determine the optimal number of clusters (K) using the Elbow Method
# You may need to customize this part
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph to determine K
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Based on the Elbow Method graph, select the optimal number of clusters (K)
optimal_k = 3  # Replace with the chosen K

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Visualize the clusters (you may need to adapt this part for your dataset)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Assign cluster labels to the original dataset
data['Cluster'] = clusters

# Interpret and analyze the clusters, and assign labels as needed

# Save or share the segmented data with the marketing team
data.to_csv('segmented_customer_data.csv', index=False)
