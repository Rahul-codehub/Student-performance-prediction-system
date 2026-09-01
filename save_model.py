import os
import pandas as pd
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Features
X = df[["study_hours", "attendance", "previous_marks"]]

# Target
y = df["final_marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Save model
joblib.dump(model, "models/student_model.pkl")

# Save evaluation results
with open("outputs/evaluation.txt", "w") as file:
    file.write("Student Performance Prediction Model Evaluation\n")
    file.write("================================================\n\n")
    file.write(f"MAE      : {mae:.4f}\n")
    file.write(f"MSE      : {mse:.4f}\n")
    file.write(f"RMSE     : {rmse:.4f}\n")
    file.write(f"R2 Score : {r2:.4f}\n")

print("Model saved successfully!")
print("Evaluation results saved successfully!")
print()
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R2 Score : {r2:.4f}")