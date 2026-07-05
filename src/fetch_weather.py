"""
Step 2: Fetch real historical daily weather data (free, no API key) and join
it to the M5 store locations.

WHY: Weather is a genuine demand driver (e.g. cold snaps -> more grocery
buying, heat waves -> more beverage sales). Adding it shows you know how to
enrich a forecasting problem with external signals, which is exactly what
real demand-planning teams do.

M5 stores are in 3 US states: CA, TX, WI. We use one representative city per
state (approx coordinates) since M5 doesn't give exact store addresses.

RUN:
    python src/fetch_weather.py
"""

import os
import time
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# One representative city per M5 state, with lat/lon
STATE_LOCATIONS = {
    "CA": {"city": "Los Angeles", "lat": 34.05, "lon": -118.24},
    "TX": {"city": "Houston", "lat": 29.76, "lon": -95.37},
    "WI": {"city": "Madison", "lat": 43.07, "lon": -89.40},
}

START_DATE = "2011-01-29"  # M5 sales history start date
END_DATE = "2016-06-19"    # M5 sales history end date

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_for_state(state, lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "America/New_York",
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame(data)
    df["state_id"] = state
    df = df.rename(columns={
        "time": "date",
        "temperature_2m_max": "temp_max_c",
        "temperature_2m_min": "temp_min_c",
        "precipitation_sum": "precipitation_mm",
    })
    return df


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    all_weather = []

    for state, loc in STATE_LOCATIONS.items():
        print(f"Fetching weather for {state} ({loc['city']})...")
        df = fetch_weather_for_state(state, loc["lat"], loc["lon"])
        all_weather.append(df)
        time.sleep(1)  # be polite to the free API

    weather_df = pd.concat(all_weather, ignore_index=True)
    out_path = os.path.join(RAW_DIR, "weather.csv")
    weather_df.to_csv(out_path, index=False)
    print(f"Saved {len(weather_df)} rows to {out_path}")
    print(weather_df.head())


if __name__ == "__main__":
    main()
