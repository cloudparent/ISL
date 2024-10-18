import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import pickle
from tqdm import tqdm  # Import tqdm for progress bar

DATA_DIR = './data'
number_of_classes = 32  # A-Z

# Load the processed feature data
X = []
y = []

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]

    for file in files:
        features = np.load(os.path.join(class_dir, file))
        X.append(features)
        y.append(j)

X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model with optimizations
rf = RandomForestClassifier(
    n_estimators=100,     # Reduced the number of trees to 100
    max_depth=10,         # Limited tree depth to 10
    random_state=42,
    n_jobs=-1             # Use all available CPU cores for parallelism
)

# Train the Random Forest model
print("Training the Random Forest model...")
rf.fit(X_train, y_train)  # Train all trees at once

# Feature selection with Recursive Feature Elimination (RFE) and progress bar
rfe = RFE(rf, n_features_to_select=20)

# Add tqdm for progress bar during RFE fitting
print("Performing Recursive Feature Elimination (RFE)...")
for _ in tqdm(range(1)):  # RFE doesn't have multiple iterations, just show one iteration of progress
    rfe.fit(X_train, y_train)

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': rfe}, f)

# Evaluate the model
y_pred = rfe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
