# required libraries
import streamlit as st
import pickle
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configure page settings - MUST BE FIRST
st.set_page_config(page_title="ChurnAI | Predict & Retain", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Inject custom CSS for advanced UI styling
st.markdown("""
    <style>
    /* Premium SaaS Dark Theme & Glassmorphism */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background Gradient - Sci-Fi Vibe */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #0b0f19 0%, #0f2027 50%, #1a1a2e 100%);
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
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #ffffff;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.8);
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
        box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.05);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 254, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.15);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 15px rgba(0, 242, 254, 0.3);
    }
    
    /* 3D Neumorphic Sidebar Radio Buttons */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(0,0,0,0.2));
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3), -4px -4px 10px rgba(255,255,255,0.03);
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        transform: translateY(-2px);
        box-shadow: 6px 6px 12px rgba(0,0,0,0.4), -6px -6px 12px rgba(255,255,255,0.04);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        box-shadow: inset 4px 4px 10px rgba(0,0,0,0.3);
        border: none;
        color: white;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    /* Pulse Animation for System Status */
    @keyframes pulse-neon {
        0% { box-shadow: 0 0 5px #28c76f; }
        50% { box-shadow: 0 0 20px #28c76f, 0 0 30px #28c76f; }
        100% { box-shadow: 0 0 5px #28c76f; }
    }
    .status-dot {
        width: 12px; height: 12px; border-radius: 50%; background: #28c76f;
        animation: pulse-neon 2s infinite;
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
    st.markdown("<h2 style='font-family:Orbitron;'>MAIN MENU</h2>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "Home", 
        "Dashboard", 
        "Prediction Engine",
        "Model Metrics",
        "Settings",
        "About System"
    ], label_visibility="collapsed")
    st.markdown("---")
    
    # New sleek system status toggle/indicator
    st.markdown("<h3 style='font-family:Orbitron; font-size:1rem; color:#cbd5e1;'>SYSTEM STATUS</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; align-items:center; gap: 10px; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; box-shadow: inset 2px 2px 5px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.05);">
        <div class="status-dot"></div>
        <span style="color: #cbd5e1; font-weight: 500; font-size:0.9rem;">Model Engine Active</span>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; font-size: 0.85rem; color: #94a3b8; border: 1px solid rgba(255,255,255,0.1);">
        <b>⚡ PRO TIP:</b><br>
        Use the Dashboard to analyze cohorts before generating specific user predictions.
    </div>
    """, unsafe_allow_html=True)


# --- PAGE: HOME ---
if page == "Home":
    st.markdown("<h1 style='font-family: Orbitron; font-weight: 900; letter-spacing: 2px; color: #00f2fe; text-shadow: 0 0 20px rgba(0,242,254,0.5);'>NEURAL CHURN PREDICTOR</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.2rem; margin-bottom: 30px;'>Turn reactive losses into proactive retention with enterprise-grade machine learning.</p>", unsafe_allow_html=True)
    
    # Hero Section Grid
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #00f2fe;">
            <h3 style="margin-top: 0; color: #fff; font-family:Orbitron;">System Overview</h3>
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
        <div class="glass-card" style="background: rgba(0,242,254,0.05); border: 1px solid rgba(0,242,254,0.2);">
            <h3 style="margin-top: 0; color: #00f2fe; font-family:Orbitron;">🚀 Quick Start Guide</h3>
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
    st.markdown("<h1 style='font-family:Orbitron; font-weight: 700;'>Data Analytics Workspace</h1>", unsafe_allow_html=True)
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
                                        color_discrete_sequence=["#00f2fe", "#ea5455"])
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
                                            color_discrete_sequence=["#00f2fe", "#43e97b", "#fa709a"])
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
                <div class="glass-card" style="border-left: 4px solid #00f2fe; background: linear-gradient(90deg, rgba(0,242,254,0.1) 0%, rgba(0,0,0,0) 100%);">
                    {ins}
                </div>
                """, unsafe_allow_html=True)


# --- PAGE: PREDICTION ENGINE ---
elif page == "Prediction Engine":
    st.markdown("<h1 style='font-family:Orbitron; font-weight: 700; color:#00f2fe;'>Prediction Engine Grid</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Simulate individual customer profiles or upload a dataset for batch AI risk assessments.</p>", unsafe_allow_html=True)

    # Mapping readable text to model numeric values
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No Internet Service": 2}

    # Advanced Tabs layout for splitting Single vs Batch Prediction
    pred_tab1, pred_tab2 = st.tabs(["👤 Single Profile Analysis", "📁 Batch Cohort Scoring"])
    
    with pred_tab1:
        # Form Card
        with st.container():
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-family:Orbitron;'>🛠️ Configure Target Parameters</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<span style='color:#cbd5e1; font-weight:500;'>Service Specifications</span>", unsafe_allow_html=True)
                tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, help="How many months has the customer stayed?")
                contract_sel = st.selectbox("Contract Type", list(contract_map.keys()))
                internet_sel = st.selectbox("Internet Service", list(internet_map.keys()))
            with col2:
                st.markdown("<span style='color:#cbd5e1; font-weight:500;'>Financial Metrics</span>", unsafe_allow_html=True)
                monthly_charges = st.slider("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=50.0, step=0.5)
                
                # Smart dynamic default for total charges
                est_total = float(monthly_charges * tenure)
                total_charges = st.slider("Total Charges ($)", min_value=0.0, max_value=10000.0, value=est_total if est_total <= 10000 else 10000.0, step=10.0)
                
            st.markdown("</div>", unsafe_allow_html=True)

        contract = contract_map[contract_sel]
        internet_service = internet_map[internet_sel]
        
        if st.button("⚡ EXECUTE NEURAL SCAN", use_container_width=True):
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
                    
                    st.markdown("<hr style='border:1px solid rgba(0,242,254,0.3); box-shadow: 0 0 10px #00f2fe;'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='font-family:Orbitron; color:#fff; text-align:center;'>SCAN RESULTS COMPILED</h3>", unsafe_allow_html=True)
                    
                    # Advanced Dashboard Results Layout
                    res_col1, res_col2, res_col3 = st.columns([1, 1, 1.2])
                    
                    with res_col2:
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
                                    {'range': [0, 40], 'color': "rgba(40, 199, 111, 0.6)"},   
                                    {'range': [40, 70], 'color': "rgba(255, 159, 67, 0.6)"},  
                                    {'range': [70, 100], 'color': "rgba(234, 84, 85, 0.8)"}  
                                ]
                            }
                        ))
                        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=280, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                    with res_col1:
                        # Gamified Radar Chart showing Customer Profile
                        categories = ['Tenure', 'Monthly Rate', 'Total Spent', 'Contract Length', 'Connectivity']
                        
                        # Normalize data for the radar polygon visual
                        norm_tenure = (tenure / 72) * 100 if tenure > 0 else 5
                        norm_monthly = (monthly_charges / 120) * 100
                        norm_total = (total_charges / 8000) * 100 if total_charges <= 8000 else 100
                        norm_contract = (contract / 2) * 100 if contract > 0 else 5
                        norm_internet = ((2 - internet_service) / 2) * 100 if internet_service < 2 else 5
                        
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[norm_tenure, norm_monthly, norm_total, norm_contract, norm_internet],
                            theta=categories,
                            fill='toself',
                            fillcolor='rgba(0, 242, 254, 0.2)',
                            line=dict(color='#00f2fe', width=2),
                            name='Target Profile'
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=False, range=[0, 100]), bgcolor='rgba(0,0,0,0)'),
                            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#cbd5e1", title={'text': "Attribute Web", 'font': {'color': '#94a3b8', 'size': 18}},
                            height=280, margin=dict(l=30, r=30, t=40, b=10)
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                    with res_col3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if pred_value == 1 or probability > 50:
                            st.markdown("""
                            <div style='background: rgba(234, 84, 85, 0.15); border: 1px solid #ea5455; border-radius: 12px; padding: 20px; box-shadow: 0 0 20px rgba(234,84,85,0.2);'>
                                <h3 style='color: #ea5455; margin-top:0; font-family:Orbitron;'>⚠️ TARGET: UNSTABLE</h3>
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
                            <div style='background: rgba(40, 199, 111, 0.15); border: 1px solid #28c76f; border-radius: 12px; padding: 20px; box-shadow: 0 0 20px rgba(40,199,111,0.2);'>
                                <h3 style='color: #28c76f; margin-top:0; font-family:Orbitron;'>✅ TARGET: STABLE</h3>
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
                    
    with pred_tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📂 Upload Cohort Data for Batch Scoring")
        st.markdown("<p style='color: #94a3b8; font-size: 0.95rem;'>Upload a CSV containing multiple customers. The engine will rapidly score everyone and provide a downloadable intelligence report.</p>", unsafe_allow_html=True)
        
        batch_file = st.file_uploader("Upload CSV File", type=["csv"], key="batch_upload")
        
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                st.markdown("##### 🔍 Data Preview:")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("⚡ EXECUTE BATCH PROTOCOL", use_container_width=True, key="batch_btn"):
                    with st.spinner("Scoring cohort through Random Forest Engine..."):
                        # Data preprocessing safely
                        required_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
                        missing_cols = [col for col in required_cols if col not in batch_df.columns]
                        
                        if missing_cols:
                            st.error(f"⚠️ Missing required columns for prediction: **{', '.join(missing_cols)}**")
                        else:
                            score_df = batch_df.copy()
                            # Ensure mappings are correctly formatted for model ingestion
                            if score_df['Contract'].dtype == 'O':
                                score_df['Contract'] = score_df['Contract'].map(contract_map).fillna(0)
                            if score_df['InternetService'].dtype == 'O':
                                net_map = {"DSL": 0, "Fiber optic": 1, "No": 2, "No Internet Service": 2}
                                score_df['InternetService'] = score_df['InternetService'].map(net_map).fillna(0)
                                
                            score_df['TotalCharges'] = pd.to_numeric(score_df['TotalCharges'], errors='coerce').fillna(0)
                            
                            X_batch = score_df[required_cols]
                            probs = model.predict_proba(X_batch)[:, 1] * 100
                            
                            batch_df['Churn_Risk_Score (%)'] = np.round(probs, 1)
                            batch_df['Risk_Level'] = ['High Risk' if p > 50 else 'Safe' for p in probs]
                            
                            st.success("✅ Batch scoring completed successfully!")
                            st.markdown("<hr style='border:1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                            
                            res_b1, res_b2 = st.columns([2, 1])
                            with res_b1:
                                st.dataframe(batch_df[['tenure', 'MonthlyCharges', 'Contract', 'Churn_Risk_Score (%)', 'Risk_Level']].head(15), use_container_width=True)
                            with res_b2:
                                high_risk_count = len(batch_df[batch_df['Risk_Level'] == 'High Risk'])
                                st.markdown(render_glass_metric("At Risk Accounts", f"{high_risk_count}"), unsafe_allow_html=True)
                                
                            csv_export = batch_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="📥 Download Scored Cohort Report (CSV)", data=csv_export, file_name="scored_cohort.csv", mime="text/csv", use_container_width=True)
                            
            except Exception as e:
                st.error(f"Error processing batch file: {e}")
                
        st.markdown("</div>", unsafe_allow_html=True)

# --- PAGE: MODEL METRICS ---
elif page == "Model Metrics":
    st.markdown("<h1 style='font-family:Orbitron; font-weight: 700; color:#00f2fe;'>Model Performance Metrics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>View backend Random Forest hyperparameters and feature importance rankings.</p>", unsafe_allow_html=True)
    st.info("Coming soon: Dynamic ROC curves, precision/recall distributions, and confusion matrices will be deployed here.")

# --- PAGE: SETTINGS ---
elif page == "Settings":
    st.markdown("<h1 style='font-family:Orbitron; font-weight: 700; color:#00f2fe;'>System Settings</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Configure your ChurnAI workspace and integrations.</p>", unsafe_allow_html=True)
    st.info("Coming soon: API key management, email retention webhooks, and custom UI color toggles.")

# --- PAGE: ABOUT ---
elif page == "About System":
    st.markdown("<h1 style='font-family:Orbitron; font-weight: 700; color:#00f2fe;'>System Architecture</h1>", unsafe_allow_html=True)
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
