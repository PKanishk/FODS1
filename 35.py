import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
data = pd.read_csv('transaction_data.csv')  # Replace with your dataset file
X = data[['TotalAmountSpent', 'FrequencyOfVisits']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Choosing the Number of Clusters (K) using the Elbow Method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, choose the optimal K value
optimal_k = 3  # Adjust this based on your analysis

# Step 3: Build the K-Means Model
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Step 4: Cluster Assignment
data['Cluster'] = kmeans.labels_

# Step 5: Interpretation and Visualization
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['TotalAmountSpent'], data['FrequencyOfVisits'], c=data['Cluster'], cmap='rainbow')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='black', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Total Amount Spent')
plt.ylabel('Frequency of Visits')
plt.legend()
plt.show()

# Step 6: Interpret the clusters
cluster_summary = data.groupby('Cluster').agg({'TotalAmountSpent': 'mean', 'FrequencyOfVisits': 'mean', 'CustomerID': 'count'})
print(cluster_summary)

# Step 7: Use the insights for marketing strategies
# Tailor marketing strategies for each customer segment based on their spending behavior.

# Step 8: Validation (optional)
# You can use additional metrics to assess the quality of your clusters.

# Step 9: Monitoring and Iteration
# Periodically update the clustering model and marketing strategies as needed.

# Step 10: Documentation and Reporting
# Document your process and communicate your findings and recommendations to stakeholders.

