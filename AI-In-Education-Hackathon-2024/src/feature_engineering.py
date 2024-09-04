import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('data/exams.csv')

# Create interaction terms between selected features
df['gender_parental_education'] = df['gender'] + "_" + df['parental level of education']

# Use one-hot encoding for the interaction terms
df = pd.get_dummies(df, columns=['gender_parental_education'], drop_first=True)

# Define features (X) and target (y)
X = pd.get_dummies(df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']], drop_first=True)
y = df[['math score', 'reading score', 'writing score']]

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train the model using the new feature set
model = LinearRegression()
model.fit(X_poly, y)

# Predict new scores using polynomial features
predictions = model.predict(X_poly)

print("Feature Engineering and Polynomial Features Applied:")
print("Predictions for math, reading, and writing scores:\n", predictions)
