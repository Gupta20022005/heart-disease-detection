# heart-disease-detection
# ❤️ Heart Disease Detection

A machine-learning project that predicts the likelihood of heart disease from 12 clinical measurements. It has a reproducible training pipeline and an interactive Streamlit web app for prediction, data exploration and model comparison.

> **Disclaimer:** Educational project only. It is not a medical device and must not be used for diagnosis.

## Features

- **Predict** – enter patient details (with readable labels and units) and get a disease probability from the best model.
- **Data Explorer** – class balance, feature distributions, categorical breakdowns and a correlation heatmap.
- **Model Performance** – metrics table, ROC curves, confusion matrices, 5-fold cross-validation and feature importance for six models.
- **Light/dark mode** – charts follow the app theme.
- **Upload your own CSV** – the app validates the columns and retrains on it.

## Project structure

```
heart_app/
├── app.py                 # Streamlit web app
├── pipeline.py            # Training pipeline (validate → split → scale → train → evaluate → save)
├── requirements.txt
├── Cardiovascular_Disease_Dataset.csv   # dataset (see "Dataset")
└── artifacts/             # created by pipeline.py (model_bundle.joblib, results.csv, summary.json)
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Train and save the models (optional – the app trains on the fly if skipped)
python pipeline.py --data Cardiovascular_Disease_Dataset.csv --out artifacts

# 2. Launch the app
streamlit run app.py
```

Streamlit 1.46+ is recommended so the charts can follow light/dark mode.

## Dataset

Mendeley *Cardiovascular Disease Dataset* – 1,000 patients, 12 features and a binary target.
The CSV must contain these exact column names (case-insensitive; `patientid` is optional):

| Column | Meaning | Values |
|---|---|---|
| `age` | Age | years |
| `gender` | Gender | 0 = female, 1 = male |
| `chestpain` | Chest pain type | 0 typical angina, 1 atypical angina, 2 non-anginal, 3 asymptomatic |
| `restingBP` | Resting blood pressure | 94–200 mm Hg |
| `serumcholestrol` | Serum cholesterol | 126–564 mg/dl |
| `fastingbloodsugar` | Fasting blood sugar > 120 mg/dl | 0 = false, 1 = true |
| `restingrelectro` | Resting ECG | 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| `maxheartrate` | Max heart rate achieved | 71–202 |
| `exerciseangia` | Exercise-induced angina | 0 = no, 1 = yes |
| `oldpeak` | ST depression | 0–6.2 |
| `slope` | Slope of peak exercise ST segment | 1 upsloping, 2 flat, 3 downsloping |
| `noofmajorvessels` | Number of major vessels | 0–3 |
| `target` | Heart disease | 0 = absent, 1 = present |

The spellings `restingelectro` and `exerciseangina` are also accepted.

## Pipeline

1. **Validate & clean** – check required columns, binary target and missing values; drop duplicates.
2. **Split** – stratified 70/30 train/test split (seed 42).
3. **Preprocess** – `MinMaxScaler` on the continuous features, inside a scikit-learn `Pipeline` (no data leakage, including during cross-validation).
4. **Train** – Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN and SVC.
5. **Evaluate** – accuracy, precision, recall, F1, ROC-AUC and MCC on the hold-out set, plus 5-fold CV ROC-AUC.
6. **Select** – the best model is chosen by CV ROC-AUC, so the test set stays untouched.
7. **Save** – models and results go to `artifacts/`.

## Results (hold-out test set)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| XGBoost (selected) | 0.983 | 0.986 | 0.999 |
| Random Forest | 0.987 | 0.988 | 0.999 |
| Logistic Regression | 0.983 | 0.986 | 0.998 |
| SVC | 0.973 | 0.977 | 0.997 |
| Decision Tree | 0.953 | 0.960 | 0.990 |
| KNN | 0.943 | 0.950 | 0.982 |

These scores are unusually high; even logistic regression reaches about 98%, which suggests the dataset is easy to separate. Treat them as a demo result, not as evidence of clinical performance.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · Plotly · Streamlit · joblib
