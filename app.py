# required libraries
import streamlit as st
import pickle
import pandas as pd


# Load the trained machine learning model
model = pickle.load(open("model.pkl", "rb"))

# create title for the web application
st.title("Customer Churn Prediction App")

# create input fields for user data
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")


# convert categorical input into numeric values
if gender == "Male":
    gender = 1
else:
    gender = 0


# create input data in dataframe format

input_data = pd.DataFrame({
    "gender":[gender],
    "tenure":[tenure],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges]
})


# add missing columns if model expects more features

expected_features = model.feature_names_in_

for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns exactly like model training
input_data = input_data[expected_features]


# create prediction button
if st.button("Predict Churn"):

    # Step 9: Make prediction using the model
    prediction = model.predict(input_data)

    # Step 10: Display result
    if prediction[0] == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer will stay")