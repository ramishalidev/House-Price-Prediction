# 🏠 Intelligent House Price Prediction System

**Advanced Python Programming - Semester Project**

## 👥 Group Members
- Muhammad Usman Rajput (450327)
- Muhammad Ramish Ali (537262)
- Malik Huzaifa Saeed (539701)

---

## 📋 Project Overview

An intelligent machine learning system that predicts house sale prices using 79+ features. Includes:
- **Jupyter Notebook** — Complete ML pipeline with 6 models
- **FastAPI Backend** — RESTful API for predictions
- **Streamlit UI** — Beautiful, unique web interface

---

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)
```bash
# Windows
run.bat

# Mac/Linux  
chmod +x run.sh && ./run.sh
```

### Option 2: Run Separately
```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start API server
uvicorn api:app --reload --port 8000

# Terminal 2: Start Streamlit UI
streamlit run app.py
```

**Access Points:**
- 🎨 **UI**: http://localhost:8501
- 📡 **API Docs**: http://localhost:8000/docs

---

## 📁 Project Structure

```
House-Price-Prediction/
├── dataset/
│   ├── train.csv              # Training data (1,460 samples)
│   ├── test.csv               # Test data (1,459 samples)
│   └── data_description.txt   # Feature descriptions
├── visualizations/            # Generated plots
├── house-price-prediction.ipynb  # Main ML notebook
├── api.py                     # FastAPI backend server
├── app.py                     # Streamlit UI
├── requirements.txt           # Dependencies
├── run.bat                    # Windows launcher
├── run.sh                     # Unix/Mac launcher
└── README.md
```

---

## 🎨 Features

### Machine Learning Models
| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization |
| Random Forest | Ensemble method |
| Gradient Boosting | Sequential boosting |
| XGBoost | Optimized gradient boosting |

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Get price prediction |
| `/neighborhoods` | GET | List neighborhoods by tier |
| `/feature-options` | GET | Get all categorical options |
| `/docs` | GET | Interactive API documentation |

### Streamlit UI Features
- 🎨 Unique glassmorphism design
- 🌈 Custom color scheme (teal, coral, gold)
- 📊 Real-time property overview
- 🏘️ Neighborhood tier indicators
- 📈 Confidence-based predictions
- 💻 Works offline (local prediction fallback)

---

## 📊 Dataset

**Source**: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

| Attribute | Value |
|-----------|-------|
| Training Samples | 1,460 |
| Test Samples | 1,459 |
| Features | 79 |
| Target | SalePrice |

---

## 📚 Technologies Used

- ✅ **Python** — Core language
- ✅ **Pandas & NumPy** — Data manipulation
- ✅ **scikit-learn & XGBoost** — ML models
- ✅ **Matplotlib & Seaborn** — Visualization
- ✅ **FastAPI** — REST API backend
- ✅ **Streamlit** — Web UI framework

---

## 📝 License

MIT License
