import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Claims Pricing Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 3px solid #38bdf8 !important;
    }
    
    /* Prediction Box Glassmorphism */
    .prediction-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        margin-top: 20px;
    }
    .prediction-amount {
        font-size: 48px;
        font-weight: 800;
        color: #10b981;
        margin: 10px 0;
    }
    .prediction-label {
        font-size: 18px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'src', 'processed', 'absolute_best_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

with st.sidebar:
    st.title("Claim Cost Estimator")
    st.markdown("---")
    st.markdown("""
    ### About This Tool
    This internal tool provides a data-driven estimate of total claim severity costs based on policyholder profiles and incident reports.
    
    ### Instructions
    1. Enter the customer's policy and demographic information.
    2. Input the specifics of the incident and vehicle.
    3. Click the prediction button to generate an estimated cost.
    
    The resulting estimate can be used to triage claims, assign adjuster resources, and identify potentially anomalous reports.
    """)

st.title("Claim Cost Underwriting Tool")
st.markdown("Enter claim parameters below to estimate the total claim severity cost.")

if model is None:
    st.error("Model not found. Run the pipeline first to generate absolute_best_model.joblib.")
    st.stop()

# use session state for demo autofill
if 'demo_loaded' not in st.session_state:
    st.session_state['demo_loaded'] = False

def load_demo_data():
    st.session_state['demo_loaded'] = True

col_a, col_b = st.columns([8, 2])
with col_b:
    st.button("Load Sample Data", on_click=load_demo_data, use_container_width=True)

with st.form("claim_form"):
    tab1, tab2, tab3, tab4 = st.tabs(["Policy & Customer", "Insured Profile", "Incident Details", "Vehicle Specs"])
    
    # Check if demo is loaded for defaults
    d = st.session_state['demo_loaded']
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        months_as_customer = c1.number_input("Months as Customer", min_value=0, max_value=1000, value=328 if d else 120)
        age = c2.number_input("Age", min_value=16, max_value=100, value=48 if d else 35)
        policy_state = c3.selectbox("Policy State", ["OH", "IN", "IL", "UNKNOWN"], index=0 if d else 0)
        
        c4, c5, c6 = st.columns(3)
        policy_csl = c4.selectbox("Policy CSL", ["250/500", "100/300", "500/1000", "UNKNOWN"], index=0 if d else 0)
        policy_deductable = c5.selectbox("Policy Deductible", [500, 1000, 2000], index=1 if d else 0)
        policy_annual_premium = c6.number_input("Annual Premium ($)", min_value=0.0, max_value=5000.0, value=1406.91 if d else 1200.0)
        
        c7, c8 = st.columns(2)
        umbrella_limit = c7.number_input("Umbrella Limit ($)", min_value=0, max_value=10000000, value=0 if d else 0)
        policy_tenure_at_incident = c8.number_input("Policy Tenure (Days)", min_value=0, max_value=15000, value=5400 if d else 1000)

    with tab2:
        c1, c2, c3 = st.columns(3)
        insured_sex = c1.selectbox("Gender", ["MALE", "FEMALE", "UNKNOWN"], index=0 if d else 0)
        insured_education_level = c2.selectbox("Education", ["MD", "PhD", "Associate", "Masters", "High School", "College", "JD", "UNKNOWN"], index=0 if d else 4)
        insured_occupation = c3.selectbox("Occupation", ["craft-repair", "machine-op-inspct", "sales", "armed-forces", "tech-support", "prof-specialty", "other-service", "priv-house-serv", "exec-managerial", "protective-serv", "transport-moving", "handlers-cleaners", "adm-clerical", "farming-fishing", "UNKNOWN"], index=0 if d else 2)
        
        c4, c5 = st.columns(2)
        capital_gains = c4.number_input("Capital Gains ($)", min_value=0, max_value=200000, value=53300 if d else 0)
        capital_loss = c5.number_input("Capital Loss ($)", min_value=-200000, max_value=0, value=0 if d else 0)
        
        c6, c7 = st.columns(2)
        insured_hobbies = c6.selectbox("Hobbies", ["sleeping", "reading", "board-games", "bungie-jumping", "base-jumping", "golf", "camping", "dancing", "skydiving", "movies", "hiking", "yachting", "paintball", "chess", "basketball", "polo", "cross-fit", "exercise", "UNKNOWN"], index=0 if d else 0)
        insured_relationship = c7.selectbox("Relationship", ["husband", "other-relative", "own-child", "unmarried", "wife", "not-in-family", "UNKNOWN"], index=0 if d else 3)

    with tab3:
        c1, c2, c3 = st.columns(3)
        incident_type = c1.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car", "Vehicle Theft", "UNKNOWN"], index=0 if d else 0)
        collision_type = c2.selectbox("Collision Type", ["Side Collision", "Rear Collision", "Front Collision", "UNKNOWN"], index=0 if d else 3)
        incident_severity = c3.selectbox("Incident Severity", ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage", "UNKNOWN"], index=0 if d else 1)
        
        c4, c5, c6 = st.columns(3)
        authorities_contacted = c4.selectbox("Authorities Contacted", ["Police", "Fire", "Other", "Ambulance", "None", "UNKNOWN"], index=0 if d else 0)
        incident_state = c5.selectbox("Incident State", ["SC", "VA", "NY", "OH", "WV", "NC", "PA", "UNKNOWN"], index=0 if d else 3)
        incident_city = c6.selectbox("Incident City", ["Columbus", "Riverwood", "Arlington", "Springfield", "Hillsdale", "Northbend", "Northbrook", "UNKNOWN"], index=0 if d else 0)
        
        c7, c8, c9 = st.columns(3)
        incident_hour_of_the_day = c7.slider("Hour of Day", 0, 23, value=5 if d else 12)
        incident_month = c8.slider("Month", 1, 12, value=1 if d else 1)
        incident_day_of_week = c9.slider("Day of Week (0=Mon)", 0, 6, value=6 if d else 0)
        
        c10, c11, c12, c13 = st.columns(4)
        property_damage = c10.selectbox("Property Damage", ["YES", "NO", "UNKNOWN"], index=0 if d else 2)
        bodily_injuries = c11.selectbox("Bodily Injuries", [0, 1, 2], index=1 if d else 0)
        witnesses = c12.selectbox("Witnesses", [0, 1, 2, 3], index=2 if d else 0)
        police_report_available = c13.selectbox("Police Report", ["YES", "NO", "UNKNOWN"], index=0 if d else 2)
        
        number_of_vehicles_involved = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=1 if d else 1)

    with tab4:
        c1, c2 = st.columns(2)
        auto_make = c1.selectbox("Auto Make", ["Saab", "Mercedes", "Dodge", "Chevrolet", "Nissan", "Ford", "Subaru", "BMW", "Toyota", "Audi", "Accura", "Volkswagen", "Jeep", "Honda", "UNKNOWN"], index=0 if d else 5)
        auto_model = c2.text_input("Auto Model", value="92x" if d else "F150")
        
        c3, c4 = st.columns(2)
        auto_year = c3.number_input("Auto Year", min_value=1990, max_value=2026, value=2004 if d else 2018)
        vehicle_age_at_incident = c4.number_input("Vehicle Age (Years)", min_value=0, max_value=50, value=11 if d else 5)

    submit_button = st.form_submit_button(label="Predict Claim Severity", use_container_width=True)

if submit_button:
    # align inputs with training columns order
    feature_columns = ['months_as_customer', 'age', 'policy_state', 'policy_csl', 'policy_deductable', 
                       'policy_annual_premium', 'umbrella_limit', 'insured_sex', 'insured_education_level', 
                       'insured_occupation', 'insured_hobbies', 'insured_relationship', 'capital-gains', 
                       'capital-loss', 'incident_type', 'collision_type', 'incident_severity', 
                       'authorities_contacted', 'incident_state', 'incident_city', 'incident_hour_of_the_day', 
                       'number_of_vehicles_involved', 'property_damage', 'bodily_injuries', 'witnesses', 
                       'police_report_available', 'auto_make', 'auto_model', 'auto_year', 
                       'policy_tenure_at_incident', 'vehicle_age_at_incident', 'incident_month', 'incident_day_of_week']
    
    input_data = [
        months_as_customer, age, policy_state, policy_csl, policy_deductable,
        policy_annual_premium, umbrella_limit, insured_sex, insured_education_level,
        insured_occupation, insured_hobbies, insured_relationship, capital_gains,
        capital_loss, incident_type, collision_type, incident_severity,
        authorities_contacted, incident_state, incident_city, incident_hour_of_the_day,
        number_of_vehicles_involved, property_damage, bodily_injuries, witnesses,
        police_report_available, auto_make, auto_model, auto_year,
        policy_tenure_at_incident, vehicle_age_at_incident, incident_month, incident_day_of_week
    ]
    
    df_input = pd.DataFrame([input_data], columns=feature_columns)
    
    # catboost requires string types for categorical features
    numerical_features = [
        'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', 
        'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'auto_year', 
        'policy_tenure_at_incident', 'vehicle_age_at_incident', 'incident_month', 'incident_day_of_week'
    ]
    categorical_features = [col for col in feature_columns if col not in numerical_features]
    
    for col in categorical_features:
        df_input[col] = df_input[col].astype(str)
        
    # model predicts log-space, need to inverse transform
    log_pred = model.predict(df_input)[0]
    actual_pred = np.expm1(log_pred)
    
    st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">Estimated Claim Cost</div>
            <div class="prediction-amount">${actual_pred:,.2f}</div>
            <div style="color: #94a3b8; font-size: 14px;">Estimated Value</div>
        </div>
    """, unsafe_allow_html=True)
    
    if actual_pred > 40000:
        st.warning("**High Severity Alert:** This claim exceeds the $40k threshold. Recommend routing to Senior Adjuster or SIU.")
    elif actual_pred < 10000:
        st.success("**Low Severity Fast-Track:** Claim is under $10k. Eligible for automated straight-through processing.")
