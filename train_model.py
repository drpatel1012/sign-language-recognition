import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

DATA_FILE = './data/hand_landmarks_az.csv'
MODEL_FILE = 'model.pkl'

def train_model():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file {DATA_FILE} not found. Please run collect_data.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # Feature columns: x0, y0, z0 ... x20, y20, z20
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes found: {np.unique(y)}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"Training Accuracy:   {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {test_acc * 100:.2f}%")
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully as {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
