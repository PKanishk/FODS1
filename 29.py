# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets  # You can replace this with loading your dataset
from sklearn.linear_model import LogisticRegression  # You can replace this with your model

# Step 2: Load the dataset (replace this with loading your dataset)
# Example: iris dataset
data = datasets.load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train your machine learning model (replace with your model training)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Ask the user to input the names of features and the target variable
feature_names = input("Enter feature names separated by spaces: ").split()
target_variable = input("Enter the target variable name: ")

# Step 6: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 7: Calculate and display common evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
