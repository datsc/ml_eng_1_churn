"""First date of writing 27.3.2026"""

## Import libraries 
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os


def load_data(path="data/churn.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()

    # Drop customer ID
    df.drop(columns=["customerID"], inplace=True) #errors="ignore")

    # Convert TotalCharges to numeric (has some spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.dropna(inplace=True)

    # Encode all object columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    return df

def train(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier(n_estimators=100, random_state=1529)
    model.fit(X_train, y_train)
    return model, X_test, y_test



def save_model(model, path="models/model.pkl"):
    print(path)
    os.makedirs("models", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")
 
 
def save_test_data(X_test, y_test, dir="models"):
    os.makedirs(dir, exist_ok=True)
    X_test.to_csv(os.path.join(dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(dir, "y_test.csv"), index=False)
    print(f"Test data saved to {dir}/X_test.csv and {dir}/y_test.csv")
 
if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    model, X_test, y_test = train(df)
    #print(model)
    save_model(model)
    save_test_data(X_test, y_test)
    print("Training complete!")