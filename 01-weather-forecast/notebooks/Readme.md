# Notebooks — Islamabad Daily Max Temperature Forecasting

This folder contains the exploratory analysis, feature engineering, and model experimentation workflow for forecasting next-day maximum temperature (`tomorrow_max`) in Islamabad (`33.68°N, 73.05°E`) using Open-Meteo historical daily data (`2016-01-01` to `2026-10-01`).

---

## Execution Order & Overview

| # | Notebook | Input | Output | Purpose |
|---|----------|-------|--------|---------|
| 1 | [`01-EDA.ipynb`](./01-EDA.ipynb) | `../data/processed/weather_daily.csv` | Visual & statistical insights | Inspect data quality, distributions, seasonality, and correlations with `tomorrow_max`. |
| 2 | [`02-feature-engineering.ipynb`](./02-feature-engineering.ipynb) | `../data/processed/weather_daily.csv` | `../data/processed/weather_features.csv` | Construct target, lag, 7-day rolling window, and cyclical calendar features; remove boundary `NaN` rows. |
| 3 | [`03-modeling.ipynb`](./03-modeling.ipynb) | `../data/processed/weather_features.csv` | Model evaluation & selection | Benchmark OLS, Ridge, Lasso, and ElasticNet against a naive persistence baseline; validate with `TimeSeriesSplit` and residual analysis. |

---

## 1. Exploratory Data Analysis (`01-EDA.ipynb`)

- **Dataset Integrity**: `3,927` daily records across `6` raw columns (`date`, `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `wind_speed_10m_max`) with `0` missing values and `0` duplicates.
- **Univariate Findings**:
  - `temperature_2m_max`: Ranges from `9.2°C` to `44.2°C` (mean `27.11°C`, std `7.26°C`).
  - `temperature_2m_min`: Ranges from `1.0°C` to `30.6°C` (mean `16.58°C`, std `7.13°C`).
  - `precipitation_sum`: Strongly right-skewed (median `0.0 mm`, max `161.3 mm`), reflecting mostly dry days punctuated by heavy rain events.
  - `wind_speed_10m_max`: Concentrated around moderate speeds (mean `12.95 km/h`, max `40.8 km/h`).
- **Bivariate & Correlation Insights (with `tomorrow_max`)**:
  - Strong positive linear correlation with `temperature_2m_max` ($r = 0.962$), `temperature_2m_mean` ($r = 0.954$), and `temperature_2m_min` ($r = 0.909$).
  - Weak direct linear correlation with `wind_speed_10m_max` ($r = 0.104$) and `precipitation_sum` ($r = -0.054$).
  - Clear annual seasonal cycle across months and day of the year.

---

## 2. Feature Engineering (`02-feature-engineering.ipynb`)

Transforms the `3,927 × 6` daily series into a `3,919 × 16` modeling dataset (`weather_features.csv`):

- **Target Variable**:
  - `tomorrow_max`: Next day's maximum temperature (`temperature_2m_max.shift(-1)`).
- **Autoregressive Lag Features**:
  - `temp_max_lag1`, `temp_max_lag2`, `temp_max_lag7`: Maximum temperatures `1`, `2`, and `7` days prior to capture persistence and short-term warming/cooling momentum.
- **Rolling Window Statistics (7-Day)**:
  - `temp_max_roll_mean_7`: 7-day rolling mean to smooth single-day weather noise and capture weekly baseline trends.
  - `temp_max_roll_std_7`: 7-day rolling standard deviation to quantify recent temperature volatility.
- **Calendar & Cyclical Seasonality Features**:
  - `month` (`1–12`) and `day_of_year` (`1–366`).
  - `sin_day_of_year` and `cos_day_of_year`: Trigonometric projection ($\sin/\cos(2\pi \cdot \text{day\_of\_year} / 365.25)$) so linear models treat late December and early January as adjacent winter days.
- **Missing Value Handling**:
  - Drops `8` boundary rows (`7` initial rows from 7-day lags/rolling windows and `1` final row from `shift(-1)`), avoiding synthetic imputation noise.

---

## 3. Modeling & Evaluation (`03-modeling.ipynb`)

- **Chronological Split**: First `80%` (`3,135` rows) for training and last `20%` (`784` rows) for testing (`shuffle=False` to prevent data leakage).
- **Feature Scaling**: `StandardScaler` fit strictly on `x_train` and applied to `x_train` and `x_test`.
- **Model Comparison on Test Set**:

| Model | Hyperparameters | Test MAE (°C) | Test RMSE (°C) | Beats Baseline? |
|-------|-----------------|---------------|----------------|-----------------|
| Naive Persistence Baseline | $\hat{y}_{t+1} = y_t$ | `1.474` | `2.032` | — |
| OLS Linear Regression | Default | `1.4223` | `1.9172` | Yes |
| Ridge Regression ($L_2$) | `alpha=10.0` | `1.4162` | `1.9185` | Yes |
| **Lasso Regression ($L_1$)** | **`alpha=0.01`** | **`1.4097`** | **`1.9213`** | **Yes (Best MAE)** |
| ElasticNet | `alpha=0.01, l1_ratio=0.9` | `1.4109` | `1.9190` | Yes |

- **Lasso Feature Selection (`alpha=0.01`)**:
  - **Eliminated (`weight = 0.0`)**: `precipitation_sum`, `temp_max_lag2`, `month`.
  - **Top Retained Drivers**: `temperature_2m_max` (`+4.9657`), `temperature_2m_mean` (`+1.4242`), `temp_max_roll_mean_7` (`+0.7842`), `cos_day_of_year` (`-0.6176`), `temperature_2m_min` (`-0.6113`).
- **Time-Series Cross-Validation (`TimeSeriesSplit`, 5 folds)**:
  - Rolling fold MAEs: `[1.2599, 1.5543, 1.3959, 1.3299, 1.4690]` °C → **Mean CV MAE: `1.4018 ± 0.1030°C`**.
- **Residual Analysis**:
  - Near-zero mean bias (`+0.0592°C`).
  - Max under-prediction: `+4.87°C`; max over-prediction: `-9.65°C` on `2025-02-19` (predicted `22.9°C` vs. actual `13.3°C` due to an abrupt cold/rain front).
