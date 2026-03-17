# libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os


# dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'telco.csv')

if not os.path.exists(csv_path):
    available_csvs = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    raise FileNotFoundError(
        f"\n\nERROR: Could not find 'telco.csv' in {base_dir}.\n"
        f"Available CSV files in this folder: {available_csvs}\n"
        "Please rename your dataset to 'telco.csv' or change the filename in this script."
    )

df = pd.read_csv(csv_path)


# data cleaning
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)


# encoding
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = le.fit_transform(df[column])


# features
X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']]
y = df["Churn"]


# test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# model
model = RandomForestClassifier()
model.fit(X_train, y_train)


# evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# saving
model_path = os.path.join(base_dir, "model.pkl")
pickle.dump(model, open(model_path, "wb"))
