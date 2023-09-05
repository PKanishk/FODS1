import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset (assuming it's in a CSV file)
data = pd.read_csv('house_data.csv')

# Select the feature and target variable
selected_feature = 'house_size'
target_variable = 'house_price'

# Bivariate Analysis
plt.scatter(data[selected_feature], data[target_variable])
plt.title(f'Bivariate Analysis: {selected_feature} vs {target_variable}')
plt.xlabel(selected_feature)
plt.ylabel(target_variable)
plt.show()

# Split the dataset into training and testing sets
X = data[[selected_feature]]
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2) Score: {r2}')

# Plot the regression line
plt.scatter(X_test, y_test, label='Test Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title(f'Linear Regression: {selected_feature} vs {target_variable}')
plt.xlabel(selected_feature)
plt.ylabel(target_variable)
plt.legend()
plt.show()
