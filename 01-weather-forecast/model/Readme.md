# Model — Production Training & Serialized Pipeline

This folder contains the final training notebook and serialized scikit-learn pipeline selected from the experiments in [`../notebooks/03-modeling.ipynb`](../notebooks/03-modeling.ipynb).

---

## Contents

| File | Description |
|------|-------------|
| [`model.ipynb`](./model.ipynb) | End-to-end training pipeline that loads engineered features, fits the `StandardScaler` + `Lasso(alpha=0.01)` pipeline, evaluates test performance, and exports the model artifact. |
| [`lasso_model.joblib`](./lasso_model.joblib) | Serialized `sklearn.pipeline.Pipeline` (`StandardScaler` → `Lasso(alpha=0.01)`) saved via `joblib.dump`. |

---

## Pipeline Specification (`model.ipynb`)

- **Input Dataset**: [`../data/processed/weather_features.csv`](../data/processed/weather_features.csv) (`3,919` daily records).
- **Target (`y`)**: `tomorrow_max` (next-day maximum temperature at 2m, in `°C`).
- **Input Features (`14` columns, in exact order)**:
  1. `temperature_2m_max`
  2. `temperature_2m_min`
  3. `temperature_2m_mean`
  4. `precipitation_sum`
  5. `wind_speed_10m_max`
  6. `temp_max_lag1`
  7. `temp_max_lag2`
  8. `temp_max_lag7`
  9. `temp_max_roll_mean_7`
  10. `temp_max_roll_std_7`
  11. `month`
  12. `day_of_year`
  13. `sin_day_of_year`
  14. `cos_day_of_year`
- **Train / Test Split**: Chronological `80/20` split (`test_size=0.2, shuffle=False`).
- **Pipeline Steps**:
  1. `scaler`: `StandardScaler()`
  2. `Lasso`: `Lasso(alpha=0.01)`

---

## Test Set Performance

| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | `1.4097 °C` |
| **$R^2$ Score** | `0.9206` (`92.06%` variance explained) |

---

## Usage: Loading & Predicting with `lasso_model.joblib`

```python
import pandas as pd
from joblib import load

# Load the trained pipeline (includes StandardScaler + Lasso)
pipeline = load("01-weather-forecast/model/lasso_model.joblib")

# Load engineered features
df = pd.read_csv("01-weather-forecast/data/processed/weather_features.csv")

feature_cols = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max",
    "temp_max_lag1", "temp_max_lag2", "temp_max_lag7",
    "temp_max_roll_mean_7", "temp_max_roll_std_7",
    "month", "day_of_year", "sin_day_of_year", "cos_day_of_year",
]

# Predict next-day maximum temperature (°C)
predictions = pipeline.predict(df[feature_cols].tail(5))
print(predictions)
```
