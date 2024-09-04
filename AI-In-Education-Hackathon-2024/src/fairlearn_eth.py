import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.reductions import GridSearch, DemographicParity, ErrorRate
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('data/exams.csv')

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
X = data.drop(columns=['math score', 'reading score', 'writing score'])
y = data[['math score', 'reading score', 'writing score']].values

# Normalize the features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Normalize the target variables
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Increased dropout rate to 20%

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network
input_size = X.shape[1]
output_size = y.shape[1]
model = NeuralNet(input_size, output_size)

# Apply Fairlearn
# Convert y to a single target by summing the scores to simplify bias mitigation
y_sum = y.sum(axis=1)

# Define demographic parity as the fairness metric
sensitive_features = data[['gender', 'race/ethnicity']]

# Logistic regression model as a surrogate for fairness mitigation
logreg = LogisticRegression()

# Fairlearn grid search with demographic parity constraint
mitigator = GridSearch(logreg, constraints=DemographicParity(), grid_size=20)
mitigator.fit(X, y_sum, sensitive_features=sensitive_features)

# Find the best predictor
best_predictor = mitigator.best_estimator_

# Use the best predictor to make predictions
y_pred_sum = best_predictor.predict(X)

# Decompose the sum prediction into individual predictions for math, reading, and writing
# We'll distribute the sum prediction proportionally based on the original y distributions
y_pred = y_pred_sum[:, None] * (y / y.sum(axis=1, keepdims=True))

# Reverse normalization on predictions
y_pred_original = scaler_y.inverse_transform(y_pred)

# Calculate residuals
residuals = data[['math score', 'reading score', 'writing score']].values - y_pred_original

# Add predictions and residuals back to the dataframe
data['predicted_math_score'] = y_pred_original[:, 0]
data['predicted_reading_score'] = y_pred_original[:, 1]
data['predicted_writing_score'] = y_pred_original[:, 2]
data['residual_math_score'] = residuals[:, 0]
data['residual_reading_score'] = residuals[:, 1]
data['residual_writing_score'] = residuals[:, 2]

# Plotting gender analysis
sns.boxplot(x='gender', y='predicted_math_score', data=data)
plt.title('Predicted Math Scores by Gender')
plt.show()

sns.boxplot(x='gender', y='predicted_reading_score', data=data)
plt.title('Predicted Reading Scores by Gender')
plt.show()

sns.boxplot(x='gender', y='predicted_writing_score', data=data)
plt.title('Predicted Writing Scores by Gender')
plt.show()

sns.boxplot(x='gender', y='residual_math_score', data=data)
plt.title('Residual Math Scores by Gender')
plt.show()

sns.boxplot(x='gender', y='residual_reading_score', data=data)
plt.title('Residual Reading Scores by Gender')
plt.show()

sns.boxplot(x='gender', y='residual_writing_score', data=data)
plt.title('Residual Writing Scores by Gender')
plt.show()

# Plotting ethnicity analysis
sns.boxplot(x='race/ethnicity', y='predicted_reading_score', data=data)
plt.title('Predicted Reading Scores by Race/Ethnicity')
plt.show()

sns.boxplot(x='race/ethnicity', y='residual_reading_score', data=data)
plt.title('Residual Reading Scores by Race/Ethnicity')
plt.show()

sns.boxplot(x='race/ethnicity', y='predicted_writing_score', data=data)
plt.title('Predicted Writing Scores by Race/Ethnicity')
plt.show()

sns.boxplot(x='race/ethnicity', y='residual_writing_score', data=data)
plt.title('Residual Writing Scores by Race/Ethnicity')
plt.show()