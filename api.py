"""
FastAPI Backend for House Price Prediction
==========================================
Uses the trained Linear Regression model from the notebook.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import pickle
import os

# Initialize FastAPI app
app = FastAPI(
    title="🏠 House Price Prediction API",
    description="Predict house prices using trained Linear Regression model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for house features (simplified for UI)
class HouseFeatures(BaseModel):
    overall_quality: int = Field(..., ge=1, le=10, description="Overall Quality (1-10)")
    overall_condition: int = Field(..., ge=1, le=10, description="Overall Condition (1-10)")
    year_built: int = Field(..., ge=1800, le=2025, description="Year Built")
    year_remod: int = Field(..., ge=1800, le=2025, description="Year Remodeled")
    lot_area: int = Field(..., ge=0, description="Lot Area (sq ft)")
    gr_liv_area: int = Field(..., ge=0, description="Above Ground Living Area (sq ft)")
    total_bsmt_sf: int = Field(default=0, ge=0, description="Total Basement Area (sq ft)")
    first_flr_sf: int = Field(..., ge=0, description="First Floor (sq ft)")
    second_flr_sf: int = Field(default=0, ge=0, description="Second Floor (sq ft)")
    bedrooms: int = Field(..., ge=0, le=10, description="Bedrooms Above Ground")
    full_bath: int = Field(..., ge=0, le=5, description="Full Bathrooms")
    half_bath: int = Field(default=0, ge=0, le=3, description="Half Bathrooms")
    kitchen_qual: str = Field(default="TA", description="Kitchen Quality")
    garage_cars: int = Field(default=0, ge=0, le=5, description="Garage Car Capacity")
    garage_area: int = Field(default=0, ge=0, description="Garage Area (sq ft)")
    fireplaces: int = Field(default=0, ge=0, le=4, description="Number of Fireplaces")
    neighborhood: str = Field(default="NAmes", description="Neighborhood")

    class Config:
        json_schema_extra = {
            "example": {
                "overall_quality": 7,
                "overall_condition": 5,
                "year_built": 2005,
                "year_remod": 2010,
                "lot_area": 8500,
                "gr_liv_area": 1800,
                "total_bsmt_sf": 1000,
                "first_flr_sf": 1000,
                "second_flr_sf": 800,
                "bedrooms": 3,
                "full_bath": 2,
                "half_bath": 1,
                "kitchen_qual": "Gd",
                "garage_cars": 2,
                "garage_area": 500,
                "fireplaces": 1,
                "neighborhood": "CollgCr"
            }
        }

# Response model
class PredictionResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    confidence: str
    model_used: str

# Simplified prediction using coefficients (fallback when model not available)
class SimplePricePredictor:
    """Fallback predictor when trained model is not available."""
    
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
    
    def predict(self, features: HouseFeatures) -> tuple:
        price = self.BASE_PRICE
        price *= self.QUALITY_WEIGHTS.get(features.overall_quality, 1.0)
        price += features.gr_liv_area * 50
        price += features.total_bsmt_sf * 30
        price += features.garage_area * 25
        price += features.garage_cars * 5000
        price += features.full_bath * 8000
        price += features.half_bath * 4000
        
        age = 2024 - features.year_built
        if age < 5: price *= 1.1
        elif age < 15: price *= 1.05
        elif age > 50: price *= 0.85
        elif age > 30: price *= 0.92
        
        price *= self.NEIGHBORHOOD_MULTIPLIERS.get(features.neighborhood, 1.0)
        price *= self.KITCHEN_QUAL_WEIGHTS.get(features.kitchen_qual, 1.0)
        price += features.fireplaces * 5000
        price += features.lot_area * 2
        
        return price, price * 0.88, price * 1.12

# Global variables for model
model = None
scaler = None
feature_names = None
fallback_predictor = SimplePricePredictor()

def load_model():
    """Load the trained model, scaler, and feature names."""
    global model, scaler, feature_names
    
    model_path = 'models/linear_regression_model.pkl'
    scaler_path = 'models/scaler.pkl'
    features_path = 'models/feature_names.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("⚠️ Model files not found. Run the notebook first to train the model.")
        return False

# Load model on startup
model_loaded = load_model()

@app.get("/")
async def root():
    """API Health Check"""
    return {
        "status": "online",
        "message": "House Price Prediction API is running!",
        "model_loaded": model_loaded,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """Predict house price based on input features."""
    
    if model is not None and scaler is not None and feature_names is not None:
        # Use trained model
        try:
            # Create a DataFrame with all features initialized to 0
            input_df = pd.DataFrame(0, index=[0], columns=feature_names)
            
            # Fill in the basic features
            input_df['OverallQual'] = features.overall_quality
            input_df['OverallCond'] = features.overall_condition
            input_df['YearBuilt'] = features.year_built
            input_df['YearRemodAdd'] = features.year_remod
            input_df['LotArea'] = np.log1p(features.lot_area)  # Log transform
            input_df['GrLivArea'] = np.log1p(features.gr_liv_area)
            input_df['TotalBsmtSF'] = np.log1p(features.total_bsmt_sf)
            input_df['1stFlrSF'] = np.log1p(features.first_flr_sf)
            input_df['2ndFlrSF'] = np.log1p(features.second_flr_sf)
            input_df['BedroomAbvGr'] = features.bedrooms
            input_df['FullBath'] = features.full_bath
            input_df['HalfBath'] = features.half_bath
            input_df['GarageCars'] = features.garage_cars
            input_df['GarageArea'] = np.log1p(features.garage_area)
            input_df['Fireplaces'] = features.fireplaces
            
            # Engineered features
            total_sf = features.total_bsmt_sf + features.first_flr_sf + features.second_flr_sf
            input_df['TotalSF'] = np.log1p(total_sf)
            input_df['HouseAge'] = 2010 - features.year_built  # Using 2010 as reference
            input_df['RemodelAge'] = 2010 - features.year_remod
            input_df['TotalBath'] = features.full_bath + 0.5 * features.half_bath
            input_df['HasGarage'] = 1 if features.garage_area > 0 else 0
            input_df['HasBsmt'] = 1 if features.total_bsmt_sf > 0 else 0
            input_df['HasFireplace'] = 1 if features.fireplaces > 0 else 0
            
            # Kitchen quality encoding
            kitchen_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
            input_df['KitchenQual'] = kitchen_map.get(features.kitchen_qual, 2)
            
            # Neighborhood one-hot encoding
            neighborhood_col = f'Neighborhood_{features.neighborhood}'
            if neighborhood_col in feature_names:
                input_df[neighborhood_col] = 1
            
            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction_log = model.predict(input_scaled)[0]
            predicted_price = np.expm1(prediction_log)  # Convert from log scale
            
            # Confidence based on quality
            if features.overall_quality >= 7:
                confidence = "High"
                range_pct = 0.10
            elif features.overall_quality >= 4:
                confidence = "Medium"
                range_pct = 0.12
            else:
                confidence = "Low"
                range_pct = 0.15
            
            return PredictionResponse(
                predicted_price=round(predicted_price, 2),
                price_range_low=round(predicted_price * (1 - range_pct), 2),
                price_range_high=round(predicted_price * (1 + range_pct), 2),
                confidence=confidence,
                model_used="Trained Linear Regression"
            )
            
        except Exception as e:
            # Fallback to simple predictor on error
            print(f"Model prediction error: {e}")
            price, low, high = fallback_predictor.predict(features)
            return PredictionResponse(
                predicted_price=round(price, 2),
                price_range_low=round(low, 2),
                price_range_high=round(high, 2),
                confidence="Medium",
                model_used="Fallback (Simple)"
            )
    else:
        # Use fallback predictor
        price, low, high = fallback_predictor.predict(features)
        confidence = "High" if features.overall_quality >= 7 else ("Medium" if features.overall_quality >= 4 else "Low")
        
        return PredictionResponse(
            predicted_price=round(price, 2),
            price_range_low=round(low, 2),
            price_range_high=round(high, 2),
            confidence=confidence,
            model_used="Fallback (Run notebook to train model)"
        )

@app.get("/model-status")
async def model_status():
    """Check if trained model is available."""
    return {
        "model_loaded": model_loaded,
        "model_type": "Linear Regression" if model_loaded else "Fallback",
        "message": "Trained model is ready!" if model_loaded else "Run the notebook to train the model."
    }

@app.get("/neighborhoods")
async def get_neighborhoods():
    """Get list of available neighborhoods."""
    return {
        "premium": ["StoneBr", "NridgHt", "NoRidge"],
        "above_average": ["Somerst", "Timber", "Veenker", "CollgCr", "Crawfor", "ClearCr", "Blmngtn"],
        "average": ["Gilbert", "NWAmes", "SawyerW", "Mitchel"],
        "below_average": ["NAmes", "NPkVill", "SWISU", "Blueste", "Sawyer", "OldTown", "Edwards"],
        "budget": ["BrkSide", "BrDale", "IDOTRR", "MeadowV"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
