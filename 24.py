import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = np.array([
    [1.2, 3.4, 2.5, 0],
    [2.0, 2.7, 1.8, 1],
    [3.6, 1.5, 2.0, 0],
])

X = data[:, :-1]
y = data[:, -1] 

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

new_patient_features = []
num_features = X_train.shape[1]
for i in range(num_features):
    feature_value = float(input(f"Enter value for feature {i+1}: "))
    new_patient_features.append(feature_value)

k = int(input("Enter the number of neighbors (k): "))

knn_classifier = KNeighborsClassifier(n_neighbors=k)

knn_classifier.fit(X_train, y_train)

new_patient_features = np.array(new_patient_features).reshape(1, -1)
predicted_label = knn_classifier.predict(new_patient_features)

if predicted_label[0] == 0:
    condition = "No medical condition"
else:
    condition = "Has the medical condition"

print(f"The patient {condition}.")
