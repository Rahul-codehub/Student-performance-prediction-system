# Student Performance Prediction System

## Project Description

The Student Performance Prediction System is an end-to-end Machine Learning project that predicts a student's final marks based on:

- Study Hours
- Attendance
- Previous Marks

The project uses Linear Regression for prediction and Flask to provide a REST API.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib

## Project Structure

student-performance-prediction/

├── app.py
├── save_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── student_data.csv
│
└── models/
    └── student_model.pkl

## Machine Learning Workflow

1. Load the student dataset.
2. Check for missing values and duplicate records.
3. Separate features and target.
4. Split the data into training and testing sets.
5. Train a Linear Regression model.
6. Evaluate the model.
7. Save the trained model.
8. Deploy the model using Flask.
9. Send student information through the `/predict` API.
10. Return the predicted final marks.

## Input Features

| Feature | Description |
|---|---|
| study_hours | Number of hours studied |
| attendance | Attendance percentage |
| previous_marks | Previous examination marks |

## Target

The target variable is:

`final_marks`

## Model Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Results on the current test split:

```text
MAE      : 0.342
MSE      : 0.184
RMSE     : 0.429
R² Score : 0.998


# How to Run
1. Install dependencies
pip install -r requirements.txt

2. Train and save the model
python save_model.py

This creates:
models/student_model.pkl

3. Start Flask
python app.py