import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

# Combine into a dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Define a simplified neural network with ReLU and Dropout
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

# Initialize the neural network, loss function, and optimizer
input_size = X.shape[1]
output_size = y.shape[1]
model = NeuralNet(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Reduced learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Set the number of epochs
num_epochs = 30  # Adjusted for potential overfitting

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
    print(f'Fold {fold + 1}')
    batch_size = 32
    # Create data loaders
    train_loader = DataLoader(dataset=TensorDataset(X_tensor[train_idx], y_tensor[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=TensorDataset(X_tensor[val_idx], y_tensor[val_idx]), batch_size=batch_size, shuffle=False)
    
    # Variable to store the minimum validation loss for the current fold
    min_val_loss = float('inf')
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
        
        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Update the minimum validation loss if the current one is lower
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        
        scheduler.step(val_loss)
    
    # Print the minimum validation loss for this fold
    print(f'Minimum Validation Loss for Fold {fold + 1}: {min_val_loss}')


# Group data by gender
males = data[data['gender'] == 1]  # Assuming 1 represents male
females = data[data['gender'] == 0]  # Assuming 0 represents female

# Get predictions for males and females
X_males = torch.tensor(scaler_X.transform(males.drop(columns=['math score', 'reading score', 'writing score'])), dtype=torch.float32)
y_males_pred = model(X_males).detach().numpy()

X_females = torch.tensor(scaler_X.transform(females.drop(columns=['math score', 'reading score', 'writing score'])), dtype=torch.float32)
y_females_pred = model(X_females).detach().numpy()

# Calculate mean predictions
mean_pred_males = y_males_pred.mean(axis=0)
mean_pred_females = y_females_pred.mean(axis=0)

print(f'Mean Predictions for Males: {mean_pred_males}')
print(f'Mean Predictions for Females: {mean_pred_females}')