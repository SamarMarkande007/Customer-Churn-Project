import streamlit as st
import requests

st.title("💳 Customer Churn Prediction")

st.write("Enter customer details:")

# Inputs
credit_score = st.number_input("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", value=50000.0)
num_products = st.number_input("Number of Products", 1, 4, 2)
has_card = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", value=50000.0)


if st.button("Predict"):

    data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card,
        "IsActiveMember": active,
        "EstimatedSalary": salary
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=data
    )

    if response.status_code == 200:
        result = response.json()

        st.subheader("Result:")
        st.write(f"Churn Probability: {result['churn_probability']:.2f}")

        if result["prediction"] == 1:
            st.error("⚠️ Customer likely to churn")
        else:
            st.success("✅ Customer not likely to churn")
    else:
        st.error("API Error")