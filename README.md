# Lifestyle Score Prediction

Professional, compact project repository for predicting a Lifestyle Score and Lifestyle Category from daily lifestyle metrics.

---

## Table of Contents
- **Project Overview**
- **Key Features**
- **Dataset**
- **Modeling Summary**
- **Installation**
- **Usage**n+- **Training & Reproducibility**
- **Repository Structure**
- **Recommendations & Next Steps**
- **License & Contact**

---

## Project Overview

This project demonstrates a full machine-learning pipeline that predicts a numerical `Lifestyle_Score` and a categorical `Lifestyle_Category` from a small set of daily lifestyle features (sleep, exercise, steps, water intake, stress, screen time, junk food). The project includes data exploration, preprocessing, model training/comparison, artifact serialization (`joblib`), and a lightweight Streamlit app for interactive predictions.

The main goals are:
- Build reliable regression (score) and classification (category) models.
- Package and save trained models and encoders for deployment.
- Provide a simple UI in `app.py` for end-user predictions.

## Key Features
- EDA: boxplots and correlation heatmaps to inspect distributions and relationships.
- Preprocessing: label encoding and standard scaling.
- Models evaluated: Linear models, tree-based, ensemble (RandomForest), kernel methods (SVR/SVC), and KNN.
- Final artifacts: `score.pkl`, `category.pkl`, and `labelencoder.pkl` (saved with `joblib`).
- Deployment: `app.py` — Streamlit-based interactive web UI.

## Dataset

Data file: `dataset/lifestyle_synthetic_dataset.csv`

Selected features used for modeling:
- `Sleep_hours`, `Workout_days_per_week`, `Workout_intensity`, `Steps_per_day`, `Water_intake_L`, `Stress_level`, `Screen_time_hrs`, `Junk_food_per_week`

Targets:
- Regression: `Lifestyle_Score` (numeric)
- Classification: `Lifestyle_Category` (categorical, label-encoded)

Notes: The dataset in this project is synthetic; use caution when generalizing to real populations.

## Modeling Summary

Models evaluated (examples):
- Regression: Linear Regression, DecisionTreeRegressor, RandomForestRegressor, SVR, KNeighborsRegressor.
- Classification: LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier.

Final choices in notebook:
- `SVR` used as `score_model` for regression.
- `SVC` used as `category_model` for classification.

Important considerations and fixes to apply:
- Ensure scalers used during training are saved (e.g., `scaler.pkl`) and applied in `app.py` before calling `predict`.
- Use `sc.transform()` on test sets (not `fit_transform`) to avoid leakage.
- Prefer `train_size=0.8` or `test_size=0.2` for clarity.

## Installation

Clone the repository and install dependencies:

```powershell
git clone <repo-url>
cd "Lifestyle Score prediction"
pip install -r requirements.txt
```

Recommended Python: 3.8+ (test with the version that matches your environment).

## Usage

Run the Streamlit app to interact with the model:

```powershell
streamlit run app.py
```

App behavior:
- Users provide feature values via sliders.
- App constructs a single-row `pandas` DataFrame and passes it to the saved models.
- The app displays the predicted numeric score and the decoded category label, plus a brief interpretation message.

Important: If models were trained on scaled data, ensure a saved `StandardScaler` or pipeline is loaded and used before prediction. The current `app.py` file loads `score.pkl` and `category.pkl` and `labelencoder.pkl` — add `scaler.pkl` (or load a saved Pipeline) to apply the same preprocessing.

## Repository Structure

Top-level files:
- `app.py` — Streamlit app for predictions.
- `lifestyle_prediction.ipynb` — exploratory notebook with training and EDA.
- `requirements.txt` — Python dependencies.
- `dataset/` — contains `lifestyle_synthetic_dataset.csv`.
- `models/` — (optional) folder to place `score.pkl`, `category.pkl`, `scaler.pkl`, and `labelencoder.pkl`.
- `Lifestyle_Score_Project_Report.txt` — generated project report.

## Recommendations & Next Steps

Short-term:
- Save and load the `StandardScaler` or a full `Pipeline` during training and use it in `app.py`.
- Fix test/train split usage and ensure transforms on test data are `transform()` only.
- Add unit tests for the prediction endpoint or app functions.

Medium-term:
- Add cross-validation and hyperparameter tuning (`GridSearchCV`/`RandomizedSearchCV`).
- Provide model explainability (SHAP) on the app or in a report.
- Add CI to run linting and tests on push.

Long-term:
- If possible, collect and incorporate real-world labeled data for better generalization.
- Add authentication and secure logging if the app will capture user inputs.

## License & Contact



Project author: Devarsh S R



