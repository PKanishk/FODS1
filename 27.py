import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load your dataset (replace 'X' with your feature matrix and 'y' with your target labels)
# Example: X, y = load_your_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values (you can customize this based on your dataset)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to predict churn for a new customer
def predict_churn(new_customer_features):
    # Preprocess the input features (scaling)
    new_customer_features = scaler.transform(np.array([new_customer_features]))
    
    # Make a prediction
    prediction = model.predict(new_customer_features)
    
    if prediction == 1:
        return "Churned"
    else:
        return "Not Churned"

# Example usage of the prediction function
new_customer_features = [usage_minutes, contract_duration, ...]  # Replace with your features
result = predict_churn(new_customer_features)
print("Predicted Churn Status:", result)
