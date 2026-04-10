# required libraries
import streamlit as st
import pickle
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

# Configure page settings - MUST BE FIRST
st.set_page_config(page_title="ChurnAI | Predict & Retain", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Inject custom CSS for advanced UI styling
st.markdown("""
    <style>
    /* Premium SaaS Dark Theme & Glassmorphism */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background Gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1a1a2e 0%, #0f2027 50%, #203a43 100%);
        color: #e2e8f0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.6) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Buttons Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(118, 75, 162, 0.5);
    }
    
    /* File Uploader styling */
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        border-bottom: 2px solid #667eea !important;
    }
    
    /* Glass Cards Content */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.1);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: -webkit-linear-gradient(45deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)


# --- HELPERS & COMPONENTS ---

def render_glass_metric(title, value):
    """Renders a sleek glassmorphism metric card."""
    return f"""
    <div class="glass-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

def generate_ai_insights(df):
    """Generates dynamic AI-like insights based on the uploaded data."""
    insights = []
    if 'Churn' in df.columns:
        churn_rate = (len(df[df['Churn'] == 'Yes']) / len(df)) * 100 if df['Churn'].dtype == 'O' else (len(df[df['Churn'] == 1]) / len(df)) * 100
        insights.append(f"🔍 **Macro Trend:** The baseline churn rate is currently at **{churn_rate:.1f}%**. Industry average for telcos sits around 20-25%.")
    
    if 'Contract' in df.columns and 'Churn' in df.columns:
        # Rough heuristic insight
        month_to_month = df[df['Contract'] == 'Month-to-month']
        if not month_to_month.empty:
            m2m_churn = (len(month_to_month[month_to_month['Churn'] == 'Yes']) / len(month_to_month)) * 100 if df['Churn'].dtype == 'O' else (len(month_to_month[month_to_month['Churn'] == 1]) / len(month_to_month)) * 100
            insights.append(f"⚠️ **Risk Segment:** Customers on **Month-to-Month contracts** have an elevated churn rate of **{m2m_churn:.1f}%**. Consider incentivizing annual upgrades.")
            
    if 'Tenure' in df.columns or 'tenure' in df.columns:
        t_col = 'Tenure' if 'Tenure' in df.columns else 'tenure'
        new_users = df[df[t_col] <= 6]
        if not new_users.empty and 'Churn' in df.columns:
            new_churn = (len(new_users[new_users['Churn'] == 'Yes']) / len(new_users)) * 100 if df['Churn'].dtype == 'O' else (len(new_users[new_users['Churn'] == 1]) / len(new_users)) * 100
            insights.append(f"📊 **Cohort Alert:** Customers in their first 6 months have a churn risk of **{new_churn:.1f}%**. Early onboarding engagement is critical.")
            
    if not insights:
        insights.append("Upload a dataset with 'Churn', 'Contract', and 'tenure' columns to unlock AI insights.")
        
    return insights


# Load the trained machine learning model
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    return pickle.load(open(model_path, "rb")), base_dir

model, base_dir = load_model()

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6105/6105565.png", width=60) # Placeholder AI Logo
    st.markdown("## **ChurnAI** Workspace")
    st.markdown("---")
    page = st.radio("Navigation", ["Home", "Dashboard", "Prediction Engine", "About System"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; font-size: 0.85rem; color: #94a3b8;">
        <b>⚡ PRO TIP:</b><br>
        Use the Dashboard to analyze cohorts before generating specific user predictions.
    </div>
    """, unsafe_allow_html=True)


# --- PAGE: HOME ---
if page == "Home":
    st.markdown("<h1 style='font-weight: 700; letter-spacing: -1px;'>Predict & Prevent Customer Churn.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.2rem; margin-bottom: 30px;'>Turn reactive losses into proactive retention with enterprise-grade machine learning.</p>", unsafe_allow_html=True)
    
    # Hero Section Grid
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top: 0; color: #fff;">Why use ChurnAI?</h3>
            <p style="color: #cbd5e1; line-height: 1.6;">
            Acquiring a new customer can cost five times more than retaining an existing one. 
            ChurnAI connects your historical telecommunications data with advanced Random Forest algorithms 
            to identify at-risk accounts <b>before</b> they cancel their subscriptions.
            </p>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><b>Analyze:</b> Deep dive into cohort distributions and financial impacts.</li>
                <li><b>Predict:</b> Generate real-time risk gauges for individual customers.</li>
                <li><b>Act:</b> Deploy targeted retention strategies powered by data.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #667eea;">
            <h3 style="margin-top: 0; color: #fff;">🚀 Quick Start</h3>
            <ol style="color: #cbd5e1; line-height: 2;">
                <li>Navigate to the <b>Dashboard</b> tab.</li>
                <li>Upload your latest customer CSV file.</li>
                <li>Review auto-generated <b>AI Insights</b>.</li>
                <li>Head to the <b>Prediction Engine</b> to score high-value accounts.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


# --- PAGE: DASHBOARD ---
elif page == "Dashboard":
    st.markdown("<h1 style='font-weight: 700;'>Data Analytics Workspace</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Analyze macro trends, view visual cohort breakdowns, and extract AI Insights.</p>", unsafe_allow_html=True)
    
    # Toolbar
    with st.container():
        uploaded_file = st.file_uploader("📂 Upload Custom Dataset (CSV)", type=["csv"], help="Expected columns: Churn, Contract, tenure, MonthlyCharges, etc.")
    
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.toast(f"Successfully loaded {len(df):,} records!", icon="✅")
        except Exception as e:
            st.error(f"Error reading custom file: {e}")
    else:
        csv_path = os.path.join(base_dir, "telco.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.markdown("<span style='color:#667eea; font-size:0.9rem;'>● Displaying default dataset (telco.csv)</span>", unsafe_allow_html=True)
        else:
            st.error("Default data file not found. Please upload a CSV.")
            
    if df is not None:
        # SaaS KPI Row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(render_glass_metric("Total Customers", f"{len(df):,}"), unsafe_allow_html=True)
        with col_m2:
            if 'Churn' in df.columns:
                churn_c = len(df[df['Churn'] == 'Yes']) if df['Churn'].dtype == 'O' else len(df[df['Churn'] == 1])
                st.markdown(render_glass_metric("Global Churn Rate", f"{(churn_c/len(df))*100:.1f}%"), unsafe_allow_html=True)
            else:
                st.markdown(render_glass_metric("Global Churn Rate", "N/A"), unsafe_allow_html=True)
        with col_m3:
            if 'MonthlyCharges' in df.columns:
                st.markdown(render_glass_metric("Avg MRR / User", f"${df['MonthlyCharges'].mean():.2f}"), unsafe_allow_html=True)
            else:
                st.markdown(render_glass_metric("Avg MRR", "N/A"), unsafe_allow_html=True)
        with col_m4:
            if 'TotalCharges' in df.columns:
                tc = pd.to_numeric(df['TotalCharges'], errors='coerce').sum()
                st.markdown(render_glass_metric("Total Lifetime Value", f"${tc/1000000:.2f}M"), unsafe_allow_html=True)
            else:
                st.markdown(render_glass_metric("Lifetime Value", "N/A"), unsafe_allow_html=True)
                
        # Main Content Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Interactive Visuals", "🧠 AI Insights & Segmentation", "🗃️ Raw Data Export"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if 'Churn' in df.columns:
                    churn_counts = df['Churn'].value_counts().reset_index()
                    churn_counts.columns = ['Churn', 'Count']
                    fig_churn = px.pie(churn_counts, values='Count', names='Churn', hole=0.6, 
                                       title="Churn Distribution", 
                                       color_discrete_sequence=["#667eea", "#ea5455"])
                    fig_churn.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
                    st.plotly_chart(fig_churn, use_container_width=True)
                else:
                    st.info("No 'Churn' column found.")
            with col_c2:
                if 'Contract' in df.columns:
                    contract_counts = df['Contract'].value_counts().reset_index()
                    contract_counts.columns = ['Contract Type', 'Customers']
                    fig_contract = px.bar(contract_counts, x='Contract Type', y='Customers',
                                          title="Cohort by Contract Type",
                                          color='Contract Type',
                                          color_discrete_sequence=["#4facfe", "#43e97b", "#fa709a"])
                    fig_contract.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", showlegend=False)
                    st.plotly_chart(fig_contract, use_container_width=True)
                else:
                    st.info("No 'Contract' column found.")
                    
        with tab3:
            st.markdown("<br>### Secure Data Viewer", unsafe_allow_html=True)
            st.dataframe(df.head(100), use_container_width=True)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Full Report (CSV)",
                               data=csv_data,
                               file_name="churn_report.csv",
                               mime="text/csv")
                               
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            insights = generate_ai_insights(df)
            for ins in insights:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #4facfe;">
                    {ins}
                </div>
                """, unsafe_allow_html=True)


# --- PAGE: PREDICTION ENGINE ---
elif page == "Prediction Engine":
    st.markdown("<h1 style='font-weight: 700;'>Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Simulate customer profiles to generate real-time AI risk assessments.</p>", unsafe_allow_html=True)

    # Mapping readable text to model numeric values
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No Internet Service": 2}

    # Form Card
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🛠️ Configure Customer Parameters")
        
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
            
        st.markdown("</div>", unsafe_allow_html=True)

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

# --- PAGE: ABOUT ---
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