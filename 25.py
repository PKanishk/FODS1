from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# User input for the new flower's features
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

# Predict the species of the new flower
new_flower_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
predicted_species = classifier.predict(new_flower_features)

# Convert the predicted species index to actual species name
species_names = iris.target_names
predicted_species_name = species_names[predicted_species[0]]

print(f"The predicted species of the new flower is: {predicted_species_name}")
