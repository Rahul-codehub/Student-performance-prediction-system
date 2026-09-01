# 🎓 Student Performance Prediction System

An end-to-end **Machine Learning and Flask REST API project** that predicts a student's final examination marks based on their **study hours, attendance percentage, and previous examination marks**.

The project demonstrates the complete machine learning lifecycle — from **data preprocessing and model training to evaluation, model persistence, and API deployment**.

---

## 📌 Project Overview

The **Student Performance Prediction System** uses **Linear Regression** to estimate a student's expected final marks.

The prediction is based on three important academic factors:

* 📚 **Study Hours** — Number of hours spent studying
* 📊 **Attendance** — Student's attendance percentage
* 📝 **Previous Marks** — Marks obtained in a previous examination

The trained machine learning model is saved using **Joblib** and exposed through a **Flask REST API**, allowing users or applications to send student information and receive predicted final marks.

---

## ✨ Key Features

* ✅ Data loading and preprocessing
* ✅ Missing-value checking
* ✅ Duplicate-record detection
* ✅ Feature and target separation
* ✅ Train-test data splitting
* ✅ Linear Regression model training
* ✅ Model performance evaluation
* ✅ MAE, MSE, RMSE, and R² metrics
* ✅ Trained model persistence using Joblib
* ✅ Flask REST API deployment
* ✅ JSON-based prediction requests
* ✅ Easy local setup and execution

---

## 🧠 Machine Learning Model

### Linear Regression

The project uses **Linear Regression** to predict the student's final marks.

The model learns the relationship between the input features:

```text
Study Hours
Attendance
Previous Marks
```

and the target variable:

```text
Final Marks
```

Conceptually, the model estimates:

```text
Final Marks = f(Study Hours, Attendance, Previous Marks)
```

The trained model is then saved as:

```text
models/student_model.pkl
```

This allows the Flask application to load the already-trained model without retraining it every time the API starts.

---

## 🔄 Machine Learning Workflow

The complete workflow followed by this project is:

```text
                ┌──────────────────────┐
                │   Student Dataset    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Data Preprocessing   │
                │ • Missing Values     │
                │ • Duplicate Records  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Feature / Target     │
                │ Separation           │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Train-Test Split     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Linear Regression    │
                │ Model Training       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Model Evaluation     │
                │ MAE / MSE / RMSE     │
                │ R² Score             │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Save Model           │
                │ student_model.pkl    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Flask REST API       │
                │      /predict        │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Predicted Final      │
                │ Marks                │
                └──────────────────────┘
```

---

## 🛠️ Technologies Used

| Technology      | Purpose                               |
| --------------- | ------------------------------------- |
| 🐍 Python       | Core programming language             |
| 🐼 Pandas       | Data manipulation and preprocessing   |
| 🔢 NumPy        | Numerical operations                  |
| 🤖 Scikit-learn | Machine learning model and evaluation |
| 🌐 Flask        | REST API development                  |
| 💾 Joblib       | Saving and loading the trained model  |

---

## 📂 Project Structure

```text
student-performance-prediction/
│
├── 📄 app.py
├── 📄 save_model.py
├── 📄 requirements.txt
├── 📄 README.md
│
├── 📁 data/
│   └── 📄 student_data.csv
│
└── 📁 models/
    └── 📦 student_model.pkl
```

### File Description

#### `app.py`

Flask application responsible for:

* Loading the trained model
* Receiving prediction requests
* Processing input data
* Returning predicted final marks

#### `save_model.py`

Machine learning training script responsible for:

* Loading the dataset
* Checking data quality
* Preparing features and target
* Splitting the dataset
* Training Linear Regression
* Evaluating the model
* Saving the trained model

#### `data/student_data.csv`

Contains the student dataset used for training and testing the machine learning model.

#### `models/student_model.pkl`

Serialized Linear Regression model generated after training.

#### `requirements.txt`

Contains the Python dependencies required to run the project.

---

## 📊 Dataset

The model uses the following input features:

| Feature          | Type      | Description                                |
| ---------------- | --------- | ------------------------------------------ |
| `study_hours`    | Numerical | Number of hours the student studies        |
| `attendance`     | Numerical | Student attendance percentage              |
| `previous_marks` | Numerical | Marks obtained in the previous examination |

### Target Variable

```text
final_marks
```

The target represents the student's predicted final examination marks.

---

## 📈 Model Evaluation

The trained Linear Regression model was evaluated using four standard regression metrics:

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

**Current result:**

```text
MAE: 0.342
```

### Mean Squared Error — MSE

Measures the average squared difference between actual and predicted values.

**Current result:**

```text
MSE: 0.184
```

### Root Mean Squared Error — RMSE

The square root of MSE, expressed in the same unit as the target variable.

**Current result:**

```text
RMSE: 0.429
```

### R² Score

Measures how well the model explains the variation in the target variable.

**Current result:**

```text
R² Score: 0.998
```

### Evaluation Summary

| Metric   |    Result |
| -------- | --------: |
| MAE      | **0.342** |
| MSE      | **0.184** |
| RMSE     | **0.429** |
| R² Score | **0.998** |

> **Note:** These results correspond to the current train-test split and dataset used in the project. Model performance can change if the dataset, preprocessing, random state, or train-test split is changed.

---

# 🚀 Getting Started

Follow the steps below to run the project locally.

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rahul-codehub/Student-performance-prediction.git
```

Navigate into the project:

```bash
cd Student-performance-prediction
```

---

## 2️⃣ Create a Virtual Environment

Creating a virtual environment is recommended to keep project dependencies isolated.

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# 🧪 Train the Machine Learning Model

Before starting the Flask API, train and save the model.

Run:

```bash
python save_model.py
```

After successful execution, the trained model will be saved as:

```text
models/student_model.pkl
```

---

# 🌐 Start the Flask API

Run:

```bash
python app.py
```

The Flask server should start locally.

```text
http://127.0.0.1:5000
```

---

# 🔮 Prediction API

The project provides a `/predict` endpoint for making predictions.

### Endpoint

```text
POST /predict
```

### Request Format

Send the student's information as JSON:

```json
{
    "study_hours": 5,
    "attendance": 85,
    "previous_marks": 78
}
```

### Example using cURL

```bash
curl -X POST http://127.0.0.1:5000/predict ^
-H "Content-Type: application/json" ^
-d "{\"study_hours\":5,\"attendance\":85,\"previous_marks\":78}"
```

### Example Response

```json
{
    "predicted_final_marks": 82.45
}
```

> The exact prediction depends on the trained model and dataset.

---

# 🔁 Complete Execution Flow

You can run the complete project using the following sequence:

```bash
# Clone repository
git clone https://github.com/Rahul-codehub/Student-performance-prediction.git

# Enter project
cd Student-performance-prediction

# Create virtual environment
python -m venv venv

# Activate environment - Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python save_model.py

# Start API
python app.py
```

The API can then be accessed at:

```text
http://127.0.0.1:5000
```

---

# 🧩 API Architecture

```text
Client
  │
  │  JSON Request
  ▼
┌──────────────────────┐
│    Flask REST API    │
│      /predict        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Load Trained Model   │
│ student_model.pkl    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Process Input        │
│ Study Hours          │
│ Attendance           │
│ Previous Marks       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Linear Regression    │
│ Prediction           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ JSON Response        │
│ Predicted Marks      │
└──────────────────────┘
```

---

# 🎯 Learning Objectives

This project demonstrates practical knowledge of:

* Machine Learning fundamentals
* Supervised learning
* Regression problems
* Linear Regression
* Data preprocessing
* Feature engineering concepts
* Train-test splitting
* Model evaluation
* Regression metrics
* Model serialization
* REST API development
* Flask
* Python project organization
* Integrating Machine Learning models with backend applications

---

# 🔮 Future Improvements

The project can be extended with several additional features:

* 📊 Add a web-based frontend
* 📈 Add prediction visualization
* 🤖 Compare multiple regression algorithms
* ⚙️ Add hyperparameter tuning
* 🧹 Improve data preprocessing
* 📉 Add residual/error analysis
* 🔐 Add API validation
* 🐳 Dockerize the application
* ☁️ Deploy the Flask API to a cloud platform
* 📚 Expand the dataset with more student records
* 🧠 Experiment with Random Forest and XGBoost
* 📊 Create an interactive student performance dashboard

---

# ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

The predictions generated by the model should not be treated as an official assessment of a student's academic performance. Real-world academic prediction systems require larger, more representative datasets and appropriate validation before being used for decision-making.

---

# 👨‍💻 Author

**Rahul Kumar**

Computer Science & Engineering

GitHub: **[Rahul-codehub](https://github.com/Rahul-codehub)**

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

### 📌 Project Summary

> **Student Performance Prediction System** is an end-to-end Machine Learning project that uses **Linear Regression** to predict final student marks from study hours, attendance, and previous marks, with the trained model deployed through a **Flask REST API**.
