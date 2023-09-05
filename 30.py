import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

# Step 1: Load the dataset
data = pd.read_csv('used_cars_dataset.csv')  # Replace 'used_cars_dataset.csv' with your dataset file

# Step 2: Preprocess the dataset
# Assuming 'mileage', 'age', 'brand', and 'engine_type' are the features, and 'price' is the target variable
X = data[['mileage', 'age', 'brand', 'engine_type']]
y = data['price']

# Encode categorical variables (brand and engine_type) using one-hot encoding
X = pd.get_dummies(X, columns=['brand', 'engine_type'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a CART model
cart_model = DecisionTreeRegressor(random_state=42)
cart_model.fit(X_train, y_train)

# Step 5: Allow user input for a new car
print("Enter the details of the new car:")
mileage = float(input("Mileage (in miles): "))
age = int(input("Age (in years): "))
brand = input("Brand: ")
engine_type = input("Engine Type: ")

# Step 6: Use the trained model to predict the price of the new car
# Encode the user input using one-hot encoding
user_input = pd.DataFrame([[mileage, age, brand, engine_type]], columns=X.columns)
user_input = pd.get_dummies(user_input, columns=['brand', 'engine_type'], drop_first=True)

predicted_price = cart_model.predict(user_input)[0]

# Step 7: Display the predicted price and decision path
print(f"Predicted Price: ${predicted_price:.2f}")

# Display the decision path (the sequence of conditions leading to the prediction)
tree_rules = export_text(cart_model, feature_names=list(X.columns))
print("\nDecision Path:")
print(tree_rules)
