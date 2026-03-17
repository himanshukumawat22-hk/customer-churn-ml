# required libraries
import streamlit as st
import pickle
import pandas as pd
import os


# Load the trained machine learning model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
model = pickle.load(open(model_path, "rb"))

# create title for the web application
st.title("Customer Churn Prediction App")

# create input fields for user data
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")
contract = st.selectbox("Contract", [0, 1, 2])
internet_service = st.selectbox("Internet Service", [0, 1, 2])


# create prediction button
if st.button("Predict Churn"):

    # Prepare the features exactly as the model expects
    df_features = pd.DataFrame([{
        'tenure': float(tenure),
        'MonthlyCharges': float(monthly_charges),
        'TotalCharges': float(total_charges),
        'Contract': int(contract),
        'InternetService': int(internet_service)
    }])

    try:
        # Make prediction directly
        prediction = model.predict(df_features)
        pred_value = int(prediction[0])
        
        # Display result
        if pred_value == 1:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer will stay")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")