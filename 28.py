import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample customer data (replace with your dataset)
customer_data = np.array([
    [100, 2, 5],
    [200, 3, 6],
    [50, 1, 2],
    [300, 5, 8],
    [150, 2, 4],
    [250, 4, 7]
])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Number of clusters (you can adjust this)
n_clusters = 3

# Create the K-Means model
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Fit the model to the customer data
kmeans.fit(scaled_data)

# User input for a new customer
new_customer_features = []
for i in range(len(customer_data[0])):
    feature = float(input(f"Enter feature {i + 1}: "))
    new_customer_features.append(feature)

# Standardize the new customer's data
scaled_new_customer = scaler.transform([new_customer_features])

# Predict the cluster for the new customer
predicted_cluster = kmeans.predict(scaled_new_customer)

# Assign the new customer to a segment
print(f"The new customer belongs to segment {predicted_cluster[0] + 1}")
