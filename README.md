# 🚀 Customer Churn Prediction (End-to-End ML Project)

## 📌 Overview

This project predicts whether a bank customer is likely to **churn (leave the bank)** using Machine Learning.

It is a **complete end-to-end ML system** including:

* Data preprocessing pipeline
* Model training & hyperparameter tuning
* Evaluation metrics
* FastAPI backend (API)
* Streamlit frontend (UI)

---

## 🎯 Problem Statement

Customer churn is a critical business problem. Identifying customers who are likely to leave helps businesses take proactive retention actions.

---

## 🧠 ML Approach

* Model: **XGBoost Classifier**
* Optimization: **Optuna (Hyperparameter tuning)**
* Evaluation focus: **Recall (important for churn detection)**
* Custom threshold tuning for better performance

---

## ⚙️ Tech Stack

| Category      | Tools                 |
| ------------- | --------------------- |
| Language      | Python                |
| ML            | Scikit-learn, XGBoost |
| Tuning        | Optuna                |
| API           | FastAPI               |
| UI            | Streamlit             |
| Visualization | Matplotlib            |
| Serialization | Joblib                |

---

## 📂 Project Structure

```
Churn_Prediction_Project/
│
├── app/                 # FastAPI + Streamlit UI
│   ├── main.py
│   ├── schema.py
│   └── ui.py
│
├── model/               # ML logic
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── pipeline.pkl     # saved model (ignored in git)
│   └── threshold.pkl
│
├── data/
│   └── raw/
│       └── IBM_customer_data.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── params.yaml          # model configuration
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 Workflow

```
EDA → Preprocessing → Training → Tuning → Evaluation → API → UI
```

---

## 🏋️‍♂️ Model Training

```bash
python -m model.train
```

✔ Saves:

* `pipeline.pkl`
* `threshold.pkl`

---

## 🔧 Hyperparameter Tuning

```bash
python -m model.tune
```

✔ Uses Optuna to find best parameters

---

## 📊 Model Evaluation

```bash
python -m model.evaluate
```

Outputs:

* Classification Report
* Confusion Matrix
* ROC AUC Score

---

## 🌐 Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

👉 Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 🎨 Run Streamlit UI

```bash
streamlit run app/ui.py
```

---

## 🧪 Sample Input (API)

```json
{
  "CreditScore": 600,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 5,
  "Balance": 50000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 70000
}
```

---

## 📈 Key Results

* ROC AUC ≈ **0.86**
* High Recall for churn class
* Threshold optimized for business needs

---

## 🧠 Key Learnings

* Handling **imbalanced datasets**
* Using **scale_pos_weight**
* Pipeline-based preprocessing (no data leakage)
* Hyperparameter tuning with Optuna
* Deploying ML models with FastAPI

---

## 🚀 Future Improvements

* Deploy on cloud (AWS / Render / Railway)
* Add MLflow for experiment tracking
* Add Docker support
* CI/CD pipeline

---

## 👨‍💻 Author

**Samar Markande**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
