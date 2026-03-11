import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dataset/PCOS_data_without_infertility.csv")

df.columns = df.columns.str.strip()

y = df["PCOS (Y/N)"]
X = df.drop("PCOS (Y/N)", axis=1)

X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "models/pcos_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model trained successfully")