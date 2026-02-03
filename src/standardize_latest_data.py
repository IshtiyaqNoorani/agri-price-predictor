import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Locate latest raw API file
# --------------------------------------------------
raw_files = sorted(Path("data/raw").glob("mandi_prices_*.csv"))

if not raw_files:
    raise FileNotFoundError("No mandi_prices_*.csv files found in data/raw")

latest_file = raw_files[-1]
print(f"Using latest file: {latest_file}")

# --------------------------------------------------
# Load raw data
# --------------------------------------------------
df = pd.read_csv(latest_file)
print(f"Rows loaded: {len(df)}")

# --------------------------------------------------
# Standardize column names
# --------------------------------------------------
column_mapping = {
    "state": "State",
    "district": "District",
    "market": "Market",
    "commodity": "Commodity",
    "arrival_date": "Arrival_Date",
    "modal_price": "Price"
}

df = df.rename(columns=column_mapping)

# --------------------------------------------------
# Keep only required columns
# --------------------------------------------------
required_columns = list(column_mapping.values())
df = df[required_columns]

# --------------------------------------------------
# Clean data types
# --------------------------------------------------
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce", dayfirst=True)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Drop rows with missing critical values
df = df.dropna(subset=["Arrival_Date", "Price", "Commodity", "State"])

# --------------------------------------------------
# Add time features (same as preprocess)
# --------------------------------------------------
df["Year"] = df["Arrival_Date"].dt.year
df["Month"] = df["Arrival_Date"].dt.month
df["Day"] = df["Arrival_Date"].dt.day
df["DayOfWeek"] = df["Arrival_Date"].dt.weekday

# --------------------------------------------------
# Save standardized data (TEMP FILE)
# --------------------------------------------------
output_path = Path("data/processed/latest_standardized.csv")
df.to_csv(output_path, index=False)

print("Standardization completed")
print(f"Saved to: {output_path}")
print(f"Final rows: {len(df)}")
print("Sample:")
print(df.head())

