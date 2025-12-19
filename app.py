"""
🏠 HomeValue AI - Smart Predicton Platform
=========================================
High-visibility, high-contrast, user-friendly UI.
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import time

# Page Configuration - Single point of control
st.set_page_config(
    page_title="HomeValue AI | Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# --- ROBUST THEME STYLING ---
st.markdown("""
<style>
    /* Clean, modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Results Card Styling */
    .prediction-container {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        border: 2px solid #4f46e5;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.1);
        text-align: center;
        margin-top: 20px;
    }

    .prediction-value {
        font-size: 4rem;
        font-weight: 800;
        color: #4f46e5;
        margin: 10px 0;
    }

    .label-text {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    /* Action Button - High Visibility */
    .stButton > button {
        width: 100%;
        background-color: #4f46e5 !important;
        color: white !important;
        padding: 15px !important;
        border-radius: 12px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.3) !important;
        margin-top: 10px !important;
    }
    
    .stButton > button:hover {
        background-color: #4338ca !important;
        transform: scale(1.02);
    }

    /* Fix Sidebar Contrast */
    [data-testid="stSidebar"] {
        padding: 20px 10px;
    }
    
    /* Ensure metric values are readable */
    [data-testid="stMetricValue"] {
        color: #4f46e5 !important;
    }

    /* Hide Streamlit branding & GitHub links */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ============== DATA & MODEL ==============
@st.cache_resource
def load_prediction_engine():
    """Load model files from models/ directory."""
    paths = {
        'model': 'models/linear_regression_model.pkl',
        'scaler': 'models/scaler.pkl',
        'features': 'models/feature_names.pkl'
    }
    if all(os.path.exists(v) for v in paths.values()):
        try:
            with open(paths['model'], 'rb') as f: m = pickle.load(f)
            with open(paths['scaler'], 'rb') as f: s = pickle.load(f)
            with open(paths['features'], 'rb') as f: names = pickle.load(f)
            return m, s, names, True
        except:
            return None, None, None, False
    return None, None, None, False

MODEL, SCALER, FEATURE_NAMES, SUCCESS = load_prediction_engine()

# Prediction logic
def predict(data: dict) -> dict:
    if SUCCESS:
        try:
            df = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
            df['OverallQual'] = data['qual']
            df['OverallCond'] = data['cond']
            df['YearBuilt'] = data['built']
            df['YearRemodAdd'] = data['remod']
            df['LotArea'] = np.log1p(data['lot'])
            df['GrLivArea'] = np.log1p(data['area'])
            df['TotalBsmtSF'] = np.log1p(data['bsmt'])
            df['1stFlrSF'] = np.log1p(data['f1'])
            df['2ndFlrSF'] = np.log1p(data['f2'])
            df['BedroomAbvGr'] = data['beds']
            df['FullBath'] = data['full_b']
            df['HalfBath'] = data['half_b']
            df['GarageCars'] = data['g_cars']
            df['GarageArea'] = np.log1p(data['g_area'])
            df['Fireplaces'] = data['fire']
            
            # Engineered
            df['TotalSF'] = np.log1p(data['bsmt'] + data['f1'] + data['f2'])
            df['HouseAge'] = 2024 - data['built']
            df['RemodelAge'] = 2024 - data['remod']
            df['TotalBath'] = data['full_b'] + 0.5 * data['half_b']
            
            k_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
            df['KitchenQual'] = k_map.get(data['kitchen'], 2)
            
            nb_col = f"Neighborhood_{data['neighborhood']}"
            if nb_col in FEATURE_NAMES:
                df[nb_col] = 1
            
            val = np.expm1(MODEL.predict(SCALER.transform(df))[0])
            return {"val": val, "trained": True}
        except:
            pass
            
    # Simple calculation for immediate demo if model missing
    base = 100000 + (data['area'] * 80) + (data['qual'] * 15000)
    return {"val": base, "trained": False}

# ============== UI LAYOUT ==============

# Main Title Section
st.title("🏠 HomeValue AI")
st.write("Professional Property Valuation using Advanced Machine Learning (Linear Regression)")

if not SUCCESS:
    st.warning("⚠️ ML Model Files not detected in /models. Please run the Jupyter Notebook to train the model. Showing estimated results currently.")
else:
    st.success("✅ Machine Learning Engine Active")

st.divider()

# Input Columns
col_main, col_result = st.columns([1.5, 1])

with col_main:
    st.subheader("📋 Property Specification")
    
    # 3x3 Grid for main inputs
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        neighborhood = st.selectbox("Neighborhood", ["CollgCr", "NAmes", "Edwards", "OldTown", "Somerst", "Gilbert", "NridgHt", "StoneBr"])
    with row1_c2:
        year_built = st.number_input("Year Built", 1900, 2024, 2005)
    with row1_c3:
        year_remod = st.number_input("Year Remodeled", 1900, 2024, 2010)
        
    row2_c1, row2_c2, row2_c3 = st.columns(3)
    with row2_c1:
        overall_qual = st.slider("Overall Quality", 1, 10, 7)
    with row2_c2:
        overall_cond = st.slider("Overall Condition", 1, 10, 5)
    with row2_c3:
        kitchen = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"], index=1)
        
    row3_c1, row3_c2, row3_c3 = st.columns(3)
    with row3_c1:
        gr_area = st.number_input("Living Area (sqft)", 500, 10000, 1800)
    with row3_c2:
        bsmt_sf = st.number_input("Basement (sqft)", 0, 5000, 1000)
    with row3_c3:
        lot_area = st.number_input("Lot Size (sqft)", 1000, 50000, 8500)

    # Secondary details in expander to keep clean
    with st.expander("➕ Additional Details (Rooms, Baths, Garage)"):
        c1, c2, c3 = st.columns(3)
        f1_sf = c1.number_input("1st Floor Area", 300, 5000, 1000)
        f2_sf = c2.number_input("2nd Floor Area", 0, 5000, 800)
        beds = c3.number_input("Bedrooms", 0, 10, 3)
        
        c4, c5, c6 = st.columns(3)
        full_b = c4.number_input("Full Bathrooms", 0, 5, 2)
        half_b = c5.number_input("Half Bathrooms", 0, 3, 1)
        fire = c6.number_input("Fireplaces", 0, 4, 1)
        
        c7, c8 = st.columns(2)
        garage_cars = c7.slider("Garage Car Capacity", 0, 5, 2)
        garage_area = c8.number_input("Garage Area (sqft)", 0, 2000, 500)

    # BIG GENERATE BUTTON IN CENTER
    predict_btn = st.button("Calculate Property Value 🔮")

with col_result:
    st.subheader("💎 Valuation Result")
    
    if predict_btn:
        inputs = {
            "neighborhood": neighborhood, "built": year_built, "remod": year_remod,
            "qual": overall_qual, "cond": overall_cond, "kitchen": kitchen,
            "area": gr_area, "bsmt": bsmt_sf, "lot": lot_area,
            "f1": f1_sf, "f2": f2_sf, "beds": beds,
            "full_b": full_b, "half_b": half_b, "fire": fire,
            "g_cars": garage_cars, "g_area": garage_area
        }
        
        with st.spinner("Processing data..."):
            time.sleep(0.5)
            res = predict(inputs)
            val = res['val']
            
            # Prediction Container
            st.markdown(f"""
                <div class="prediction-container">
                    <p class="label-text">ESTIMATED MARKET VALUE</p>
                    <p class="prediction-value">${val:,.0f}</p>
                    <p style="color:#64748b; font-size: 0.8rem;">
                        {"Standard Regression Analysis" if res['trained'] else "Geometric Estimation"}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence Metrics
            st.divider()
            m_c1, m_c2 = st.columns(2)
            m_c1.metric("Low Estimate", f"${val * 0.93:,.0f}")
            m_c2.metric("High Estimate", f"${val * 1.07:,.0f}")
            
            # Advice
            st.info("💡 Pro Tip: Increasing the overall quality rating to 8+ could significantly boost market value.")

    else:
        st.info("Your valuation will appear here once you click the calculate button.")
        
        # Static property overview
        st.markdown("---")
        st.markdown("### 📊 Live Property Summary")
        st.metric("Subject Area", f"{gr_area:,} ft²")
        st.metric("Subject Quality", f"{overall_qual}/10")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem;">
    HomeValue AI &bull; Built for Advanced Python Project &bull; Usman, Ramish & Malik
</div>
""", unsafe_allow_html=True)
