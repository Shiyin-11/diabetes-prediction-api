# Diabetes Prediction API

An end-to-end machine learning API project for diabetes risk prediction using FastAPI and Random Forest classification.

## Project Overview

This project demonstrates a complete machine learning workflow, including:

- Healthcare data preprocessing
- Exploratory data analysis (EDA)
- Machine learning model training
- Model evaluation and serialization
- REST API development with FastAPI
- Real-time prediction serving

The application predicts diabetes risk based on patient health metrics using a trained Random Forest classifier.

---

## Tech Stack

- Python
- pandas
- scikit-learn
- FastAPI
- Uvicorn
- Jupyter Notebook
- Joblib

---

## Machine Learning Workflow

1. Data preprocessing and cleaning
2. Missing value handling
3. Feature engineering and scaling
4. Train/test split
5. Logistic Regression baseline model
6. Random Forest model training
7. Model evaluation using accuracy and classification metrics
8. Model serialization using Joblib

---

## API Endpoints

### GET /

Health check endpoint.

### POST /predict

Predict diabetes risk using patient health data.

Example request:

```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 79,
  "BMI": 25.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 33
}
```

Example response:

```json
{
  "prediction": "Low Diabetes Risk",
  "probability": 0.11
}
```

---

## Project Structure

```bash
diabetes-prediction-api/
│
├── app/
│   └── main.py
├── data/
│   └── diabetes.csv
├── models/
│   └── diabetes_model.pkl
├── notebooks/
│   └── 01_data_exploration.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shiyin-11/diabetes-prediction-api.git
```

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

---

## API Documentation

After starting the server, open:

```bash
http://127.0.0.1:8001/docs
```

to access the interactive Swagger API documentation.

## Model Performance

Random Forest Accuracy: **86%**

The model was evaluated using classification metrics including precision, recall, and F1-score on a held-out test dataset.

---

## Future Improvements

- Add model deployment with Docker
- Integrate database support
- Add frontend dashboard
- Implement CI/CD pipeline
- Experiment with XGBoost and deep learning models