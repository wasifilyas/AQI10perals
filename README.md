# 🌫️ AQI Predictor — Karachi Air Quality Forecasting Station

A real-time Air Quality Index (AQI) forecasting system for Karachi, Pakistan. It ingests hourly weather and air quality data, trains ML models daily, and serves a **3-day AQI forecast** through a polished Streamlit dashboard — all automated via GitHub Actions.

---

##  Features

| Feature | Description |
|---|---|
| **Live AQI Display** | Current AQI reading with color-coded category badge and spectrum gauge |
| **3-Day Forecast** | Daily-average AQI predictions at 24h, 48h, and 72h horizons |
| **SHAP Explainability** | Feature importance breakdown showing *why* the model made each prediction |
| **Health Alerts** | Automatic guidance when forecasted AQI enters unhealthy ranges |
| **EDA Dashboard** | Embedded 2-year exploratory analysis (time series, correlations, distributions) |
| **Accuracy Tracker** | Live prediction-vs-actual logging to monitor model performance over time |
| **Automated Pipelines** | Hourly data ingestion + daily model retraining via GitHub Actions cron jobs |

---

##  Architecture

```
┌─────────────────────┐    hourly cron     ┌──────────────────────┐
│   Open-Meteo APIs   │ ─────────────────► │   feature_pipeline   │
│  (Weather + AQI)    │                    │  (ingest & engineer) │
└─────────────────────┘                    └──────────┬───────────┘
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │   Supabase DB  │
                                              │  (aqi_features │
                                              │  model_registry│
                                              │  prediction_log)│
                                              └───────┬───────┘
                                                      │
                              daily cron              │
                         ┌────────────────────────────┘
                         ▼                            │
              ┌─────────────────────┐                 │
              │  training_pipeline  │                 │
              │  (regression +      │                 │
              │   classification)   │                 │
              └────────┬────────────┘                 │
                       │ uploads models               │
                       ▼                              │
              ┌─────────────────────┐                 │
              │  Supabase Storage   │                 │
              │  (model .joblib)    │                 │
              └─────────────────────┘                 │
                                                      │
                                                      ▼
                                            ┌──────────────────┐
                                            │     app.py       │
                                            │  (Streamlit UI)  │
                                            │  Live dashboard  │
                                            └──────────────────┘
```

---

##  Project Structure

```
aqi-predictor/
├── app.py                        # Streamlit dashboard (main UI)
├── feature_pipeline.py           # Hourly data ingestion & feature engineering
├── training_pipeline.py          # Daily regression model training (VotingRegressor)
├── classification_pipeline.py    # Daily classifier training (AQI category prediction)
├── eda_analysis.py               # Exploratory data analysis script
├── eda_output/                   # Pre-generated EDA charts (7 plots)
│   ├── 01_aqi_timeseries.png
│   ├── 02_monthly_seasonality.png
│   ├── 03_hourly_pattern.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_pm25_vs_aqi.png
│   ├── 06_windspeed_vs_aqi.png
│   └── 07_category_distribution.png
├── .github/workflows/
│   ├── feature_pipeline.yml      # Runs every hour (cron: '0 * * * *')
│   └── training_pipeline.yml     # Runs daily at 3 AM UTC (cron: '0 3 * * *')
├── requirements.txt              # Python dependencies
├── .env                          # Local environment variables (not committed)
└── .gitignore
```

---

##  ML Pipeline

### Data Sources
- **Weather:** [Open-Meteo Archive API](https://open-meteo.com/) — temperature, humidity, wind speed/direction, pressure, precipitation, boundary layer height
- **Air Quality:** [Open-Meteo Air Quality API](https://open-meteo.com/) — PM2.5, PM10, CO, NO₂, SO₂, O₃, US AQI

### Feature Engineering (24 features)
- **Rolling aggregates:** PM2.5 & AQI rolling means (6h, 24h) and rolling std (24h)
- **Cyclical encoding:** hour, month, day-of-year, wind direction → sin/cos pairs
- **Log transforms:** `log1p(PM2.5)`, `log1p(PM10)`
- **Dispersion index:** `PM2.5 / (wind_speed + 1)`
- **Future weather:** forecasted temperature, humidity, wind speed, pressure averaged over a 24h window
- **Lag features:** `aqi_lag_24h`

### Models
| Task | Model | Details |
|---|---|---|
| **Regression** (AQI value) | `VotingRegressor` | Ensemble of Ridge, RandomForest, and HistGradientBoosting with `GridSearchCV` + `TimeSeriesSplit` |
| **Classification** (AQI category) | `HistGradientBoostingClassifier` | Tuned via `GridSearchCV` + `TimeSeriesSplit`; predicts one of 6 EPA categories |

Both models are wrapped in scikit-learn `Pipeline` objects (imputation → scaling → model) and serialized with `joblib`.

### AQI Categories
| AQI Range | Category | Color |
|---|---|---|
| 0–50 | Good | 
| 51–100 | Moderate | 
| 101–150 | Unhealthy (Sensitive) | 
| 151–200 | Unhealthy | 
| 201–300 | Very Unhealthy | 
| 301+ | Hazardous | 

---

##  Setup & Installation

### Prerequisites
- Python 3.13+
- A [Supabase](https://supabase.com/) project with the following tables:
  - `aqi_features` — hourly feature store
  - `model_registry` — model metadata & active flag
  - `prediction_log` — prediction tracking for accuracy monitoring
- Supabase Storage bucket: `models`

### 1. Clone & Install

```bash
git clone https://github.com/wasifilyas/AQI10perals.git
cd aqi-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-key
CITY_NAME=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011
```

### 3. Run the Pipelines

```bash
# Ingest data & engineer features
python feature_pipeline.py

# Train regression models
python training_pipeline.py

# Train classification models
python classification_pipeline.py

# (Optional) Generate EDA plots
python eda_analysis.py
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

---

## 🔄 Automation (GitHub Actions)

| Workflow | Schedule | What it does |
|---|---|---|
| **Feature Pipeline** | Every hour (`0 * * * *`) | Fetches latest weather + AQI data, engineers features, upserts to Supabase |
| **Training Pipeline** | Daily at 3 AM UTC (`0 3 * * *`) | Retrains regression + classification models, uploads to Supabase Storage, updates model registry |

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase API key |
| `CITY_NAME` | Target city name (e.g. `Karachi`) |
| `CITY_LAT` | City latitude |
| `CITY_LON` | City longitude |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, Plotly, custom CSS (glassmorphism, animations) |
| **ML** | scikit-learn, SHAP |
| **Data** | pandas, NumPy, matplotlib, seaborn |
| **Database** | Supabase (PostgreSQL + Storage) |
| **APIs** | Open-Meteo (Weather & Air Quality) |
| **CI/CD** | GitHub Actions |
| **Serialization** | joblib |

---

##  Team

**AQI 10perals**

---

## 📄 License

This project is for educational and research purposes.
