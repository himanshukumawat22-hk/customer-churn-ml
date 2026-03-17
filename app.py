# required libraries
import streamlit as st
import pickle
import pandas as pd
import os


# Load the trained machine learning model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Prediction", "About"])

if page == "Home":
    st.title("Customer Churn AI Assistant 🤖")
    st.write("### Welcome to the Telecommunications Churn Predictor!")
    st.write("""
    This application helps telecommunication businesses identify customers who are highly likely to leave (churn). 
    By identifying these customers early, companies can offer targeted promotions and retain their business.
    
    **How to use this app:**
    1. Go to the **Prediction** page from the sidebar.
    2. Enter the customer's billing and service information.
    3. Click Predict to see if the customer is at risk of churning.
    """)
    st.info("👈 Select a page from the sidebar to continue")

elif page == "Dashboard":
    st.title("Data Analytics Dashboard 📊")
    st.write("A quick look at our historical telecommunications dataset.")
    
    csv_path = os.path.join(base_dir, "telco.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        st.subheader("Overall Customer Churn")
        st.bar_chart(df['Churn'].value_counts(), color="#ff4b4b")
        
        st.subheader("Contract Types Distribution")
        st.bar_chart(df['Contract'].value_counts(), color="#4b8bff")
    else:
        st.warning("Historical data file (telco.csv) not found for visualizations.")

elif page == "Prediction":
    st.title("Predict Customer Churn")
    st.write("Enter the customer details below to predict their likelihood of churning.")

    # Mapping readable text to model numeric values
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No Internet Service": 2}

    # create input fields for user data in columns for better layout
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0)
    
    with col2:
        contract_sel = st.selectbox("Contract Type", list(contract_map.keys()))
        internet_sel = st.selectbox("Internet Service", list(internet_map.keys()))

    contract = contract_map[contract_sel]
    internet_service = internet_map[internet_sel]

    # create prediction button
    if st.button("Predict Churn", use_container_width=True):
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
            probability = model.predict_proba(df_features)[0][1] * 100 # Get percentage of churn
            pred_value = int(prediction[0])
            
            # Visual Probability Meter
            st.subheader("Prediction Results")
            st.metric("Churn Risk Probability", f"{probability:.1f}%")
            st.progress(int(probability)) # Shows a loading-style bar
            
            # Display result
            if pred_value == 1:
                st.error(f"⚠️ High Risk: This customer is highly likely to churn.")
            else:
                st.success(f"✅ Safe: This customer is likely to stay.")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

elif page == "About":
    st.title("About This Project")
    st.write("""
    **Machine Learning Model:** Random Forest Classifier
    
    **Features used:** 
    * Tenure
    * Monthly Charges
    * Total Charges
    * Contract Type 
    * Internet Service Type
    
    This tool is designed to provide actionable insights for customer retention teams.
    """)