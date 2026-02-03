import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# CONFIG (EDIT ONLY API KEY)
# --------------------------------------------------
API_KEY = "579b464db66ec23bdd0000017f87866626954ca25d3f6ae36297c396"
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

# Save location
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Filename with date
today_str = datetime.today().strftime("%Y-%m-%d")
OUTPUT_FILE = RAW_DATA_DIR / f"mandi_prices_{today_str}.csv"

# --------------------------------------------------
# API PARAMETERS
# --------------------------------------------------
params = {
    "api-key": API_KEY,
    "format": "json",
    "limit": 10000   # safe upper limit
}

# --------------------------------------------------
# FETCH DATA
# --------------------------------------------------
print("Fetching latest mandi price data...")

response = requests.get(BASE_URL, params=params)

if response.status_code != 200:
    raise Exception(f"API request failed with status {response.status_code}")

data = response.json()

records = data.get("records", [])

if not records:
    raise Exception("No records received from API")

# --------------------------------------------------
# SAVE TO CSV
# --------------------------------------------------
df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False)

print(f" Data fetched successfully")
print(f" Saved to: {OUTPUT_FILE}")
print(f" Total rows: {len(df)}")
print("Sample data:")
print(df.head())

