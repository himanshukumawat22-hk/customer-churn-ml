# required libraries
import streamlit as st
import pickle
import pandas as pd
import os

# Configure page settings - MUST BE FIRST
st.set_page_config(page_title="Customer Churn AI", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

# Inject custom CSS for advanced UI styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1f77b4;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 242, 254, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained machine learning model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Sidebar Navigation
st.sidebar.title("🔮 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Prediction", "About"])
st.sidebar.markdown("---")
st.sidebar.info("Upload data, analyze historical trends, and predict churn using AI.")

if page == "Home":
    st.title("Customer Churn AI Assistant 🤖")
    st.markdown("### Welcome to the Next-Gen Telecommunications Churn Predictor!")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Why use this app?")
        st.write("""
        This powerful tool helps telecommunication businesses identify customers who are highly likely to leave (churn). 
        By identifying these customers early, companies can intervene, offer targeted promotions, and retain their business.
        """)
    with col2:
        st.info("""
        **🚀 Quick Start Guide:**
        1. Go to the **Prediction** tab via the sidebar.
        2. Enter the customer's billing & service info.
        3. Click Predict to generate a real-time AI risk assessment.
        """)
        
    with st.expander("📊 Analyze custom historical data?"):
        st.write("Navigate to the **Dashboard** page. There, you can upload your own customized CSV file to uncover metrics, evaluate feature distributions, and browse your raw customer data securely.")

elif page == "Dashboard":
    st.title("Data Analytics Dashboard 📊")
    st.write("Explore historical customer data or upload your own dataset to analyze churn patterns.")
    
    uploaded_file = st.file_uploader("Upload custom customer data (CSV format)", type=["csv"], help="Your data should preferably contain columns like 'Churn', 'Contract', etc. to match standard visuals.")
    
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded custom dataset with {len(df):,} records!")
        except Exception as e:
            st.error(f"Error reading custom file: {e}")
    else:
        csv_path = os.path.join(base_dir, "telco.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.info("Currently displaying default dataset (`telco.csv`). Upload a file above to visualize custom data.")
        else:
            st.warning("Historical data file (telco.csv) not found. Please upload a CSV file to continue.")
            
    if df is not None:
        st.markdown("---")
        # Advanced UI Tabs for organized dashboard
        tab1, tab2, tab3 = st.tabs(["📈 Overview Metrics", "📊 Visualizations", "🗃️ Data Explorer"])
        
        with tab1:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Customers Recorded", f"{len(df):,}")
            if 'Churn' in df.columns:
                # Check mapping safely (whether the data is encoded to 0/1 or raw text Yes/No)
                churn_count = len(df[df['Churn'] == 'Yes']) if df['Churn'].dtype == 'O' else len(df[df['Churn'] == 1])
                churn_rate = (churn_count / len(df)) * 100
                col_m2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
            if 'MonthlyCharges' in df.columns:
                avg_charge = df['MonthlyCharges'].mean()
                col_m3.metric("Avg Monthly Revenue Per User", f"${avg_charge:.2f}")
                
        with tab2:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if 'Churn' in df.columns:
                    st.subheader("Customer Churn Distribution")
                    st.bar_chart(df['Churn'].value_counts(), color="#ff4b4b")
                else:
                    st.warning("Column 'Churn' missing from uploaded data.")
            with col_c2:
                if 'Contract' in df.columns:
                    st.subheader("Contract Types")
                    st.bar_chart(df['Contract'].value_counts(), color="#4b8bff")
                else:
                    st.warning("Column 'Contract' missing from uploaded data.")
                    
        with tab3:
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(100), use_container_width=True)

elif page == "Prediction":
    st.title("Predict Customer Churn 🔮")
    st.write("Enter the customer's billing and service details below to evaluate their risk profile.")

    # Mapping readable text to model numeric values
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No Internet Service": 2}

    # Advanced layout card for inputs
    with st.container():
        st.markdown("### 📋 Customer Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, value=12, help="How many months has the customer stayed?")
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        with col3:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
        
        col4, col5 = st.columns(2)
        with col4:
            contract_sel = st.selectbox("Contract Type", list(contract_map.keys()))
        with col5:
            internet_sel = st.selectbox("Internet Service", list(internet_map.keys()))

    contract = contract_map[contract_sel]
    internet_service = internet_map[internet_sel]
    
    if st.button("🚀 Execute Risk Analysis", use_container_width=True):
        with st.spinner("Connecting to Random Forest Engine..."):
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
                
                st.markdown("<hr style='border:1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                
                # Advanced Dashboard Results Layout
                res_col1, res_col2 = st.columns([1.5, 1])
                
                with res_col1:
                    # Plotly Gauge Chart for Probability
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability,
                        number = {'suffix': "%", 'font': {'color': 'white'}},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "AI Risk Probability", 'font': {'color': '#94a3b8', 'size': 18}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "rgba(255,255,255,0.3)"},
                            'bgcolor': "rgba(0,0,0,0.2)",
                            'borderwidth': 2,
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, 30], 'color': "#28c76f"},   # Green
                                {'range': [30, 70], 'color': "#ff9f43"},  # Orange
                                {'range': [70, 100], 'color': "#ea5455"}  # Red
                            ]
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with res_col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if pred_value == 1 or probability > 50:
                        st.markdown("""
                        <div style='background: rgba(234, 84, 85, 0.15); border: 1px solid #ea5455; border-radius: 12px; padding: 20px;'>
                            <h3 style='color: #ea5455; margin-top:0;'>⚠️ High Risk Profile</h3>
                            <p style='color: #cbd5e1; font-size: 0.95rem;'>Patterns strongly align with historical churn data.</p>
                            <hr style='border: 0.5px solid rgba(234,84,85,0.3);'>
                            <b style='color: #fff;'>Action Engine Suggests:</b>
                            <ul style='color: #cbd5e1; font-size: 0.9rem;'>
                                <li>Trigger automated retention email sequence.</li>
                                <li>Offer an immediate 10% discount on an annual contract upgrade.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background: rgba(40, 199, 111, 0.15); border: 1px solid #28c76f; border-radius: 12px; padding: 20px;'>
                            <h3 style='color: #28c76f; margin-top:0;'>✅ Safe Profile</h3>
                            <p style='color: #cbd5e1; font-size: 0.95rem;'>Customer behavior suggests loyalty and stable revenue.</p>
                            <hr style='border: 0.5px solid rgba(40,199,111,0.3);'>
                            <b style='color: #fff;'>Action Engine Suggests:</b>
                            <ul style='color: #cbd5e1; font-size: 0.9rem;'>
                                <li>Maintain standard engagement protocols.</li>
                                <li>Flag for potential cross-selling opportunities (e.g., family plans).</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error generating prediction: {e}")

elif page == "About System":
    st.markdown("<h1 style='font-weight: 700;'>System Architecture</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("""
    <div class="glass-card">
        <p>This robust application is engineered to grant telecommunications businesses a strategic advantage by proactively identifying churn threats.</p>
    
        <h3>🧠 Machine Learning Engine</h3>
        <p><b>Algorithm:</b> <code>Random Forest Classifier</code></p>
    
        <h3>🔑 Core Features Analyzed:</h3>
        <ul>
            <li><b>Tenure:</b> Customer longevity and loyalty scope.</li>
            <li><b>Financials:</b> Monthly Charges vs. Total Charges to map spending tolerance.</li>
            <li><b>Contract Lifecycle:</b> Analyzes Month-to-Month vs. Annual commitment weights.</li>
            <li><b>Service Infrastructure:</b> Factoring in specific dependencies (DSL, Fiber, etc.).</li>
        </ul>
    
        <hr style="border:1px solid rgba(255,255,255,0.1)">
        <p><b>Mission Statement:</b> <i>To empower customer retention teams to turn reactive losses into proactive conversions.</i></p>
    </div>
    """, unsafe_allow_html=True)