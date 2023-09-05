import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load your dataset (assuming you have a CSV file named 'housing_data.csv')
data = pd.read_csv('housing_data.csv')

# Step 2: Preprocess the dataset (select relevant features and target variable)
X = data[['area', 'num_bedrooms']]  # Replace with actual feature columns
y = data['price']                    # Replace with the actual target column

# Step 3: Split the dataset into training and testing sets (optional)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train a linear regression model
model = LinearRegression()
model.fit(X, y)  # Use X_train and y_train if you have a train-test split

# Step 5: Take user input for the features of the new house
area = float(input("Enter the area of the new house (in square feet): "))
num_bedrooms = int(input("Enter the number of bedrooms in the new house: "))

# Step 6: Use the trained model to predict the price of the new house
new_house_features = np.array([[area, num_bedrooms]])
predicted_price = model.predict(new_house_features)

# Step 7: Display the predicted price to the user
print(f"The predicted price of the new house is ${predicted_price[0]:,.2f}")
