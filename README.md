# Heart Disease Predictor

A machine learning web application that predicts the likelihood of heart disease based on clinical parameters. Built with Python, scikit-learn, and Streamlit.

---

## Project Overview

This project trains five classification models on the UCI Heart Disease dataset and deploys the best-performing model (SVM) as an interactive web app. Users input their clinical details and receive an instant risk assessment.

---

## Tech Stack

- **Language:** Python 3.12.0
- **ML Library:** scikit-learn
- **Web Framework:** Streamlit
- **Data Handling:** pandas, numpy
- **Model Persistence:** joblib

---

## Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| K-Nearest Neighbors | Distance-based classifier |
| Naive Bayes | Probabilistic approach |
| Decision Tree | Rule-based splits |
| SVM | Best performer, deployed |

---

## Features Used

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Sex | Biological sex (M/F) |
| ChestPainType | ATA / NAP / TA / ASY |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl |
| RestingECG | Resting ECG results |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y/N) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of peak exercise ST segment |

---

## Project Structure

```
heart-disease-predictor/
│
├── frontend                # Streamlit web app
|    ├──app.py     
├── train.py                # Model training script
|   |──main.py               
├── csv file                # Dataset
|   |──heart.csv              
│
├── models/
│   ├── svm_heart.pkl       # Saved SVM model
│   ├── scaler.pkl          # Saved scaler
│   └── column.pkl          # Saved encoded column names
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/ManavCodingspace/Health-Disease-Predicator.git
cd Heart-Disease-Predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**

Model files (`.pkl`) are not included in this repo. Run the training script to generate them locally:
```bash
python train.py
```
This will create a `models/` folder with `svm_heart.pkl`, `scaler.pkl`, and `column.pkl`.

**4. Run the app**
```bash
streamlit run app.py
```

---

## Author

**Manav**  
