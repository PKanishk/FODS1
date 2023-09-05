# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset (assuming you have a CSV file with your data)
data = pd.read_csv('car_data.csv')  # Replace 'car_data.csv' with your dataset file name

# Select the features and target variable
selected_features = ['engine_size', 'horsepower', 'fuel_efficiency', 'other_features']  # Replace with your chosen features
X = data[selected_features]
y = data['price']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")

# Visualize the model's performance
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

# Get the coefficients and intercept of the linear regression model
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:")
for feature, coef in zip(selected_features, coefficients):
    print(f"{feature}: {coef}")

print(f"Intercept: {intercept}")
