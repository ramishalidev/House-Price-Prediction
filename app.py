"""
🏠 House Price Prediction - Streamlit UI
=========================================
A beautiful, unique interface for predicting house prices.
Color Scheme: Deep Teal, Warm Coral, Soft Gold, Cream

Run with: streamlit run app.py
"""

import streamlit as st
import requests
import json
from typing import Optional

# Page Configuration
st.set_page_config(
    page_title="HomeValue AI | Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for unique styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Root Variables - Unique Color Scheme */
    :root {
        --deep-teal: #1a535c;
        --ocean-teal: #2d6a78;
        --warm-coral: #ff6b6b;
        --soft-gold: #f4a261;
        --cream: #fefae0;
        --sage: #4ecdc4;
        --charcoal: #2d3436;
        --light-gray: #f8f9fa;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stApp {
        background: linear-gradient(180deg, #f8f9fa 0%, #fff 50%, #fefae0 100%);
    }
    
    /* Header Styling */
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a535c 0%, #4ecdc4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #636e72;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Card Styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(26, 83, 92, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a535c 0%, #2d6a78 100%);
        border-radius: 24px;
        padding: 2.5rem;
        color: white;
        box-shadow: 0 20px 60px rgba(26, 83, 92, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(78, 205, 196, 0.2) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .price-display {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 700;
        color: #4ecdc4;
        text-shadow: 0 2px 20px rgba(78, 205, 196, 0.3);
        margin: 1rem 0;
    }
    
    .price-label {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 500;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 1rem;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #4ecdc4 0%, #2d6a78 100%);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f4a261 0%, #e76f51 100%);
        color: white;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    
    /* Range Display */
    .range-display {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .range-item {
        text-align: center;
    }
    
    .range-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .range-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: white;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #1a535c;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4ecdc4;
        display: inline-block;
    }
    
    /* Input Labels */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #2d3436 !important;
        font-size: 0.9rem !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a535c 0%, #2d6a78 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    section[data-testid="stSidebar"] label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2.5rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 8px 30px rgba(255, 107, 107, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(255, 107, 107, 0.4);
    }
    
    /* Feature Summary Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .feature-item {
        background: linear-gradient(135deg, #fefae0 0%, #fff 100%);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(26, 83, 92, 0.1);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(26, 83, 92, 0.1);
    }
    
    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.25rem;
        color: #1a535c;
        font-weight: 600;
    }
    
    .feature-label {
        font-size: 0.75rem;
        color: #636e72;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Neighborhood Tiers */
    .tier-premium { color: #f4a261; }
    .tier-above { color: #4ecdc4; }
    .tier-average { color: #1a535c; }
    .tier-below { color: #636e72; }
    .tier-budget { color: #ff6b6b; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.6s ease-out forwards;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    /* Divider */
    .custom-divider {
        height: 4px;
        background: linear-gradient(90deg, #4ecdc4, #1a535c, #ff6b6b);
        border-radius: 2px;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

def check_api_connection():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_price(features: dict) -> Optional[dict]:
    """Call API to get prediction"""
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def local_predict(features: dict) -> dict:
    """Local prediction when API is not available"""
    # Simplified prediction logic (same as API)
    BASE_PRICE = 80000
    
    QUALITY_WEIGHTS = {
        1: 0.5, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.9,
        6: 1.0, 7: 1.15, 8: 1.35, 9: 1.55, 10: 1.8
    }
    
    NEIGHBORHOOD_MULTIPLIERS = {
        "StoneBr": 1.4, "NridgHt": 1.35, "NoRidge": 1.3,
        "Somerst": 1.15, "Timber": 1.1, "Veenker": 1.1,
        "CollgCr": 1.05, "Crawfor": 1.05, "ClearCr": 1.05,
        "Gilbert": 1.0, "NWAmes": 0.95, "SawyerW": 0.95,
        "Mitchel": 0.9, "NAmes": 0.85, "NPkVill": 0.85,
        "SWISU": 0.8, "Blueste": 0.8, "Sawyer": 0.8,
        "OldTown": 0.75, "Edwards": 0.75, "BrkSide": 0.7,
        "BrDale": 0.65, "IDOTRR": 0.65, "MeadowV": 0.6,
        "Blmngtn": 1.05
    }
    
    KITCHEN_QUAL_WEIGHTS = {"Ex": 1.15, "Gd": 1.05, "TA": 1.0, "Fa": 0.9, "Po": 0.8}
    
    price = BASE_PRICE
    quality = features.get("overall_quality", 5)
    price *= QUALITY_WEIGHTS.get(quality, 1.0)
    price += features.get("gr_liv_area", 1500) * 50
    price += features.get("total_bsmt_sf", 0) * 30
    price += features.get("garage_area", 0) * 25
    price += features.get("garage_cars", 0) * 5000
    price += features.get("full_bath", 1) * 8000
    price += features.get("half_bath", 0) * 4000
    
    age = 2024 - features.get("year_built", 2000)
    if age < 5: price *= 1.1
    elif age < 15: price *= 1.05
    elif age > 50: price *= 0.85
    elif age > 30: price *= 0.92
    
    neighborhood = features.get("neighborhood", "NAmes")
    price *= NEIGHBORHOOD_MULTIPLIERS.get(neighborhood, 1.0)
    
    kitchen = features.get("kitchen_qual", "TA")
    price *= KITCHEN_QUAL_WEIGHTS.get(kitchen, 1.0)
    
    price += features.get("fireplaces", 0) * 5000
    price += features.get("lot_area", 8000) * 2
    
    confidence = "High" if quality >= 7 else ("Medium" if quality >= 4 else "Low")
    
    return {
        "predicted_price": round(price, 2),
        "price_range_low": round(price * 0.88, 2),
        "price_range_high": round(price * 1.12, 2),
        "confidence": confidence,
        "features_summary": {
            "quality_rating": f"{quality}/10",
            "living_area": f"{features.get('gr_liv_area', 1500):,} sq ft",
            "bedrooms": features.get("bedrooms", 3),
            "bathrooms": f"{features.get('full_bath', 1)}.{features.get('half_bath', 0)}",
            "neighborhood": neighborhood
        }
    }

# ============== MAIN UI ==============

# Hero Section
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 class="hero-title">🏠 HomeValue AI</h1>
    <p class="hero-subtitle">Discover your property's worth with intelligent price prediction</p>
</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

# Sidebar - Property Configuration
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.2); margin-bottom: 1.5rem;">
        <h2 style="color: #4ecdc4; font-family: 'Playfair Display', serif; margin: 0;">Property Details</h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin: 0.5rem 0 0 0;">Configure your home features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Location
    st.markdown("##### 📍 Location")
    neighborhoods = [
        "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", 
        "Gilbert", "Sawyer", "NWAmes", "SawyerW", "Mitchel", "BrkSide",
        "Crawfor", "IDOTRR", "Timber", "NoRidge", "StoneBr", "SWISU",
        "ClearCr", "MeadowV", "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste"
    ]
    neighborhood = st.selectbox("Neighborhood", neighborhoods, index=1, 
                                 help="Location significantly impacts value")
    
    st.markdown("---")
    
    # Quality & Condition
    st.markdown("##### ⭐ Quality & Condition")
    overall_quality = st.slider("Overall Quality", 1, 10, 7, 
                                help="Rate the overall material and finish (1=Poor, 10=Excellent)")
    overall_condition = st.slider("Overall Condition", 1, 10, 5,
                                  help="Rate the overall condition (1=Poor, 10=Excellent)")
    kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=1,
                                format_func=lambda x: {"Ex": "Excellent", "Gd": "Good", "TA": "Average", "Fa": "Fair", "Po": "Poor"}[x])
    
    st.markdown("---")
    
    # Size
    st.markdown("##### 📐 Size & Area")
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 10000, 1800, step=100)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 6000, 1000, step=100)
    lot_area = st.number_input("Lot Area (sq ft)", 1000, 100000, 8500, step=500)
    first_flr_sf = st.number_input("First Floor (sq ft)", 300, 6000, 1000, step=100)
    second_flr_sf = st.number_input("Second Floor (sq ft)", 0, 4000, 800, step=100)
    
    st.markdown("---")
    
    # Rooms
    st.markdown("##### 🛏️ Rooms")
    col1, col2 = st.columns(2)
    with col1:
        bedrooms = st.number_input("Bedrooms", 0, 10, 3)
        full_bath = st.number_input("Full Baths", 0, 5, 2)
    with col2:
        half_bath = st.number_input("Half Baths", 0, 3, 1)
        fireplaces = st.number_input("Fireplaces", 0, 4, 1)
    
    st.markdown("---")
    
    # Garage
    st.markdown("##### 🚗 Garage")
    garage_types = ["Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "None"]
    garage_type = st.selectbox("Garage Type", garage_types, index=0)
    garage_cars = st.slider("Car Capacity", 0, 5, 2)
    garage_area = st.number_input("Garage Area (sq ft)", 0, 2000, 500, step=50)
    
    st.markdown("---")
    
    # Year Info
    st.markdown("##### 📅 Age & Updates")
    year_built = st.number_input("Year Built", 1800, 2024, 2005)
    year_remod = st.number_input("Year Remodeled", 1800, 2024, 2010)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h3 class="section-header">Property Overview</h3>', unsafe_allow_html=True)
    
    # Feature summary cards
    st.markdown(f"""
    <div class="glass-card">
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-icon">⭐</div>
                <div class="feature-value">{overall_quality}/10</div>
                <div class="feature-label">Quality</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📐</div>
                <div class="feature-value">{gr_liv_area:,}</div>
                <div class="feature-label">Sq Ft Living</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🛏️</div>
                <div class="feature-value">{bedrooms}</div>
                <div class="feature-label">Bedrooms</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🚿</div>
                <div class="feature-value">{full_bath}.{half_bath}</div>
                <div class="feature-label">Bathrooms</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🚗</div>
                <div class="feature-value">{garage_cars}</div>
                <div class="feature-label">Garage Cars</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🏠</div>
                <div class="feature-value">{2024 - year_built}</div>
                <div class="feature-label">Years Old</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Building info
    bldg_type = st.selectbox("Building Type", ["1Fam", "TwnhsE", "Duplex", "Twnhs", "2fmCon"], 
                             format_func=lambda x: {"1Fam": "Single Family", "TwnhsE": "Townhouse End", 
                                                    "Duplex": "Duplex", "Twnhs": "Townhouse", "2fmCon": "Two Family"}[x])
    house_style = st.selectbox("House Style", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"],
                               format_func=lambda x: {"1Story": "One Story", "2Story": "Two Story", 
                                                      "1.5Fin": "1.5 Story Finished", "SLvl": "Split Level", "SFoyer": "Split Foyer"}[x])
    pool_area = st.number_input("Pool Area (sq ft)", 0, 1000, 0, step=50)

with col2:
    st.markdown('<h3 class="section-header">📍 Neighborhood</h3>', unsafe_allow_html=True)
    
    # Neighborhood tier indicator
    premium = ["StoneBr", "NridgHt", "NoRidge"]
    above_avg = ["Somerst", "Timber", "Veenker", "CollgCr", "Crawfor", "ClearCr", "Blmngtn"]
    average = ["Gilbert", "NWAmes", "SawyerW", "Mitchel"]
    
    if neighborhood in premium:
        tier = "Premium"
        tier_color = "#f4a261"
        tier_desc = "Top-tier location with highest property values"
    elif neighborhood in above_avg:
        tier = "Above Average"
        tier_color = "#4ecdc4"
        tier_desc = "Desirable area with strong market value"
    elif neighborhood in average:
        tier = "Average"
        tier_color = "#1a535c"
        tier_desc = "Standard market pricing"
    else:
        tier = "Value"
        tier_color = "#636e72"
        tier_desc = "Affordable options with growth potential"
    
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏘️</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: #1a535c; font-family: 'Playfair Display', serif;">{neighborhood}</div>
        <div style="display: inline-block; padding: 0.3rem 1rem; background: {tier_color}; color: white; border-radius: 20px; font-size: 0.8rem; margin-top: 0.5rem; font-weight: 600;">{tier}</div>
        <p style="color: #636e72; font-size: 0.85rem; margin-top: 1rem;">{tier_desc}</p>
    </div>
    """, unsafe_allow_html=True)

# Predict Button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("🔮 Calculate Home Value", use_container_width=True)

# Prediction Result
if predict_clicked:
    # Prepare features
    features = {
        "overall_quality": overall_quality,
        "overall_condition": overall_condition,
        "year_built": year_built,
        "year_remod": year_remod,
        "lot_area": lot_area,
        "gr_liv_area": gr_liv_area,
        "total_bsmt_sf": total_bsmt_sf,
        "first_flr_sf": first_flr_sf,
        "second_flr_sf": second_flr_sf,
        "bedrooms": bedrooms,
        "full_bath": full_bath,
        "half_bath": half_bath,
        "kitchen_qual": kitchen_qual,
        "garage_cars": garage_cars,
        "garage_area": garage_area,
        "garage_type": garage_type,
        "fireplaces": fireplaces,
        "pool_area": pool_area,
        "neighborhood": neighborhood,
        "bldg_type": bldg_type,
        "house_style": house_style
    }
    
    with st.spinner("🏠 Analyzing property value..."):
        # Try API first, fallback to local prediction
        api_online = check_api_connection()
        if api_online:
            result = predict_price(features)
        else:
            result = local_predict(features)
        
        if result:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Confidence class
            conf_class = f"confidence-{result['confidence'].lower()}"
            
            # Display prediction card
            st.markdown(f"""
            <div class="prediction-card animate-in">
                <div class="price-label">Estimated Market Value</div>
                <div class="price-display">${result['predicted_price']:,.0f}</div>
                <div class="confidence-badge {conf_class}">{result['confidence']} Confidence</div>
                
                <div class="range-display">
                    <div class="range-item">
                        <div class="range-label">Low Estimate</div>
                        <div class="range-value">${result['price_range_low']:,.0f}</div>
                    </div>
                    <div class="range-item">
                        <div class="range-label">High Estimate</div>
                        <div class="range-value">${result['price_range_high']:,.0f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show data source
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <span style="color: #636e72; font-size: 0.8rem;">
                    {"🌐 Prediction from API server" if api_online else "💻 Local prediction (API offline)"}
                </span>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="custom-divider" style="margin-top: 3rem;"></div>
<div style="text-align: center; padding: 1rem; color: #636e72;">
    <p style="font-size: 0.85rem; margin: 0;">
        <strong>HomeValue AI</strong> — Intelligent House Price Prediction System
    </p>
    <p style="font-size: 0.75rem; color: #aaa; margin-top: 0.5rem;">
        Built with ❤️ using Machine Learning • For educational purposes
    </p>
</div>
""", unsafe_allow_html=True)
