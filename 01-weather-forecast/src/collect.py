import json
from pathlib import Path

import requests
import pandas as pd 

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":  33.68,
    "longitude": 73.05,
    "start_date": "2016-01-01",
    "end_date": "2026-10-01",
    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
    "timezone": "auto"
}

response = requests.get(url, params=params)
print("Status code:", response.status_code)
if response.status_code != 200:
    print("Error:", response.text)
    exit()

data = response.json()

with open(Path(__file__).resolve().parent.parent / "data" / "raw" / "weather_raw.json", "w") as f:
    json.dump(data, f)

df = pd.DataFrame(data["daily"])
df = df.rename(columns={"time": "date"})  

df.to_csv(Path(__file__).resolve().parent.parent / "data" / "processed" / "weather_daily.csv", index=False)




