import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
DAILY_PATH = Path("data/processed/cleaned_data.csv")
WEEKLY_PATH = Path("data/processed/weekly_data.csv")

# --------------------------------------------------
# Load daily data
# --------------------------------------------------
if not DAILY_PATH.exists():
    raise FileNotFoundError("cleaned_data.csv not found")

df = pd.read_csv(DAILY_PATH)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

print("Daily rows:", len(df))

# --------------------------------------------------
# Create week identifier
# --------------------------------------------------
df["YearWeek"] = df["Arrival_Date"].dt.to_period("W").astype(str)

# --------------------------------------------------
# Aggregate to weekly level
# --------------------------------------------------
weekly_df = (
    df.groupby(["YearWeek", "State", "Commodity"], as_index=False)
    .agg(
        Price=("Price", "mean"),
        Year=("Year", "first"),
        Month=("Month", "first"),
        DayOfWeek=("DayOfWeek", "mean"),
    )
)

# --------------------------------------------------
# Save weekly dataset
# --------------------------------------------------
weekly_df.to_csv(WEEKLY_PATH, index=False)

print("Weekly dataset created")
print("Weekly rows:", len(weekly_df))
print("Sample:")
print(weekly_df.head())

