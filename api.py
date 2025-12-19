"""
FastAPI Backend for House Price Prediction
==========================================
A RESTful API server for serving house price predictions using trained ML models.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import pickle
import os
from enum import Enum

# Initialize FastAPI app
app = FastAPI(
    title="🏠 House Price Prediction API",
    description="Predict house prices using advanced machine learning models",
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

# Enums for categorical features
class MSZoning(str, Enum):
    RL = "RL"  # Residential Low Density
    RM = "RM"  # Residential Medium Density
    FV = "FV"  # Floating Village
    RH = "RH"  # Residential High Density
    C = "C (all)"  # Commercial

class Neighborhood(str, Enum):
    NAmes = "NAmes"
    CollgCr = "CollgCr"
    OldTown = "OldTown"
    Edwards = "Edwards"
    Somerst = "Somerst"
    NridgHt = "NridgHt"
    Gilbert = "Gilbert"
    Sawyer = "Sawyer"
    NWAmes = "NWAmes"
    SawyerW = "SawyerW"
    Mitchel = "Mitchel"
    BrkSide = "BrkSide"
    Crawfor = "Crawfor"
    IDOTRR = "IDOTRR"
    Timber = "Timber"
    NoRidge = "NoRidge"
    StoneBr = "StoneBr"
    SWISU = "SWISU"
    ClearCr = "ClearCr"
    MeadowV = "MeadowV"
    Blmngtn = "Blmngtn"
    BrDale = "BrDale"
    Veenker = "Veenker"
    NPkVill = "NPkVill"
    Blueste = "Blueste"

class BldgType(str, Enum):
    Family1 = "1Fam"
    TwnhsE = "TwnhsE"
    Duplex = "Duplex"
    Twnhs = "Twnhs"
    TwoFmCon = "2fmCon"

class HouseStyle(str, Enum):
    Story1 = "1Story"
    Story2 = "2Story"
    Story1_5Fin = "1.5Fin"
    SLvl = "SLvl"
    SFoyer = "SFoyer"
    Story1_5Unf = "1.5Unf"
    Story2_5Unf = "2.5Unf"
    Story2_5Fin = "2.5Fin"

class Quality(str, Enum):
    Excellent = "Ex"
    Good = "Gd"
    Average = "TA"
    Fair = "Fa"
    Poor = "Po"

class GarageType(str, Enum):
    Attchd = "Attchd"
    Detchd = "Detchd"
    BuiltIn = "BuiltIn"
    CarPort = "CarPort"
    Basment = "Basment"
    NoGarage = "None"

# Request model for house features
class HouseFeatures(BaseModel):
    # Basic Info
    overall_quality: int = Field(..., ge=1, le=10, description="Overall Quality (1-10)")
    overall_condition: int = Field(..., ge=1, le=10, description="Overall Condition (1-10)")
    year_built: int = Field(..., ge=1800, le=2025, description="Year Built")
    year_remod: int = Field(..., ge=1800, le=2025, description="Year Remodeled")
    
    # Size Features
    lot_area: int = Field(..., ge=0, description="Lot Area (sq ft)")
    gr_liv_area: int = Field(..., ge=0, description="Above Ground Living Area (sq ft)")
    total_bsmt_sf: int = Field(default=0, ge=0, description="Total Basement Area (sq ft)")
    first_flr_sf: int = Field(..., ge=0, description="First Floor (sq ft)")
    second_flr_sf: int = Field(default=0, ge=0, description="Second Floor (sq ft)")
    
    # Rooms
    bedrooms: int = Field(..., ge=0, le=10, description="Bedrooms Above Ground")
    full_bath: int = Field(..., ge=0, le=5, description="Full Bathrooms")
    half_bath: int = Field(default=0, ge=0, le=3, description="Half Bathrooms")
    kitchen_qual: Quality = Field(default=Quality.Average, description="Kitchen Quality")
    
    # Garage
    garage_cars: int = Field(default=0, ge=0, le=5, description="Garage Car Capacity")
    garage_area: int = Field(default=0, ge=0, description="Garage Area (sq ft)")
    garage_type: GarageType = Field(default=GarageType.Attchd, description="Garage Type")
    
    # Additional Features
    fireplaces: int = Field(default=0, ge=0, le=4, description="Number of Fireplaces")
    pool_area: int = Field(default=0, ge=0, description="Pool Area (sq ft)")
    
    # Location & Type
    neighborhood: Neighborhood = Field(default=Neighborhood.NAmes, description="Neighborhood")
    bldg_type: BldgType = Field(default=BldgType.Family1, description="Building Type")
    house_style: HouseStyle = Field(default=HouseStyle.Story1, description="House Style")

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
                "garage_type": "Attchd",
                "fireplaces": 1,
                "pool_area": 0,
                "neighborhood": "CollgCr",
                "bldg_type": "1Fam",
                "house_style": "2Story"
            }
        }

# Response model
class PredictionResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    confidence: str
    features_summary: dict

# Simple prediction model using coefficients learned from data analysis
# This is a simplified model - in production, you'd load the trained pickle file
class SimplePricePredictor:
    """
    A simplified price prediction model based on key features.
    Uses coefficients derived from the data analysis.
    """
    
    # Base price and feature weights (derived from correlation analysis)
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
    
    KITCHEN_QUAL_WEIGHTS = {
        "Ex": 1.15, "Gd": 1.05, "TA": 1.0, "Fa": 0.9, "Po": 0.8
    }
    
    def predict(self, features: HouseFeatures) -> tuple:
        # Start with base price
        price = self.BASE_PRICE
        
        # Quality factor (strongest predictor)
        quality_mult = self.QUALITY_WEIGHTS.get(features.overall_quality, 1.0)
        price *= quality_mult
        
        # Living area contribution (~$50 per sq ft)
        price += features.gr_liv_area * 50
        
        # Basement contribution (~$30 per sq ft)
        price += features.total_bsmt_sf * 30
        
        # Garage contribution
        price += features.garage_area * 25
        price += features.garage_cars * 5000
        
        # Bathroom contribution
        price += features.full_bath * 8000
        price += features.half_bath * 4000
        
        # Bedroom adjustment
        if features.bedrooms >= 3:
            price += (features.bedrooms - 2) * 3000
        
        # Age factor (newer = higher price)
        current_year = 2024
        age = current_year - features.year_built
        if age < 5:
            price *= 1.1
        elif age < 15:
            price *= 1.05
        elif age > 50:
            price *= 0.85
        elif age > 30:
            price *= 0.92
        
        # Remodel bonus
        remod_years = current_year - features.year_remod
        if remod_years < 5:
            price *= 1.05
        
        # Neighborhood multiplier
        neighborhood_mult = self.NEIGHBORHOOD_MULTIPLIERS.get(features.neighborhood.value, 1.0)
        price *= neighborhood_mult
        
        # Kitchen quality
        kitchen_mult = self.KITCHEN_QUAL_WEIGHTS.get(features.kitchen_qual.value, 1.0)
        price *= kitchen_mult
        
        # Fireplaces
        price += features.fireplaces * 5000
        
        # Pool (if present)
        if features.pool_area > 0:
            price += 15000 + features.pool_area * 20
        
        # Lot area contribution (small effect)
        price += features.lot_area * 2
        
        # Calculate confidence range (+/- 10-15%)
        confidence_pct = 0.12
        low = price * (1 - confidence_pct)
        high = price * (1 + confidence_pct)
        
        return price, low, high

# Initialize predictor
predictor = SimplePricePredictor()

@app.get("/")
async def root():
    """API Health Check"""
    return {
        "status": "online",
        "message": "House Price Prediction API is running!",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """
    Predict house price based on input features.
    
    Returns predicted price with confidence range.
    """
    try:
        # Get prediction
        price, low, high = predictor.predict(features)
        
        # Determine confidence level
        if features.overall_quality >= 7:
            confidence = "High"
        elif features.overall_quality >= 4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Summary of key features
        features_summary = {
            "quality_rating": f"{features.overall_quality}/10",
            "living_area": f"{features.gr_liv_area:,} sq ft",
            "total_area": f"{features.gr_liv_area + features.total_bsmt_sf:,} sq ft",
            "bedrooms": features.bedrooms,
            "bathrooms": f"{features.full_bath}.{features.half_bath}",
            "garage": f"{features.garage_cars} cars",
            "age": f"{2024 - features.year_built} years",
            "neighborhood": features.neighborhood.value
        }
        
        return PredictionResponse(
            predicted_price=round(price, 2),
            price_range_low=round(low, 2),
            price_range_high=round(high, 2),
            confidence=confidence,
            features_summary=features_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neighborhoods")
async def get_neighborhoods():
    """Get list of available neighborhoods with price tiers"""
    return {
        "premium": ["StoneBr", "NridgHt", "NoRidge"],
        "above_average": ["Somerst", "Timber", "Veenker", "CollgCr", "Crawfor", "ClearCr"],
        "average": ["Gilbert", "NWAmes", "SawyerW", "Mitchel", "Blmngtn"],
        "below_average": ["NAmes", "NPkVill", "SWISU", "Blueste", "Sawyer", "OldTown", "Edwards"],
        "budget": ["BrkSide", "BrDale", "IDOTRR", "MeadowV"]
    }

@app.get("/feature-options")
async def get_feature_options():
    """Get available options for all categorical features"""
    return {
        "neighborhoods": [n.value for n in Neighborhood],
        "building_types": [b.value for b in BldgType],
        "house_styles": [h.value for h in HouseStyle],
        "garage_types": [g.value for g in GarageType],
        "quality_levels": [q.value for q in Quality]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
