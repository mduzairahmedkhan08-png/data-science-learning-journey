# Diabetes Prediction using Logistic Regression

## Project Overview

This project predicts whether a patient is likely to have diabetes using a Logistic Regression machine learning model. The model is trained on the Pima Indians Diabetes Dataset and deployed using Streamlit.

---

## Dataset Features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target Variable

- 0 → No Diabetes
- 1 → Diabetes

---

## Machine Learning Workflow

1. Import Libraries
2. Load Dataset
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Outlier Detection
6. Outlier Treatment
7. Multicollinearity Check (VIF)
8. Train-Test Split
9. Feature Scaling
10. Logistic Regression Model
11. Model Evaluation
12. Model Serialization
13. Streamlit Deployment

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Streamlit
- Joblib

---

## Project Structure

```
Diabetes_Prediction/

├── app.py
├── diabetes.csv
├── logistic_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Author

Praveen Guvvala

Graduate in Computer Science and Engineering (AI & ML)

Aspiring Data Scientist | Machine Learning Enthusiast
