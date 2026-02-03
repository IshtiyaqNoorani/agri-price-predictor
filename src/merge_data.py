import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
HISTORICAL_PATH = Path("data/processed/cleaned_data.csv")
LATEST_PATH = Path("data/processed/latest_standardized.csv")
OUTPUT_PATH = Path("data/processed/cleaned_data.csv")  # overwrite safely

# --------------------------------------------------
# Load data
# --------------------------------------------------
if not HISTORICAL_PATH.exists():
    raise FileNotFoundError("Historical cleaned_data.csv not found")

if not LATEST_PATH.exists():
    raise FileNotFoundError("latest_standardized.csv not found")

historical_df = pd.read_csv(HISTORICAL_PATH)
latest_df = pd.read_csv(LATEST_PATH)

print("Historical rows:", len(historical_df))
print("Latest rows:", len(latest_df))

# --------------------------------------------------
# Combine
# --------------------------------------------------
combined_df = pd.concat([historical_df, latest_df], ignore_index=True)

# --------------------------------------------------
# Remove duplicates
# --------------------------------------------------
combined_df["Arrival_Date"] = pd.to_datetime(combined_df["Arrival_Date"])

combined_df = combined_df.sort_values("Arrival_Date")

combined_df = combined_df.drop_duplicates(
    subset=["Arrival_Date", "State", "Commodity"],
    keep="last"
)

# --------------------------------------------------
# Save merged dataset
# --------------------------------------------------
combined_df = combined_df.reset_index(drop=True)
combined_df.to_csv(OUTPUT_PATH, index=False)

print("Merge completed successfully")
print("Final rows:", len(combined_df))
print("Latest date in dataset:", combined_df["Arrival_Date"].max())

