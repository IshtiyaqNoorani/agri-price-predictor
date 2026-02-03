import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/mandi_prices.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_data.csv")


def load_raw_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError("mandi_prices.csv not found in data/raw/")
    return pd.read_csv(RAW_DATA_PATH)


def preprocess_data(df):
    # Clean column names
    df.columns = (
        df.columns
        .str.replace("x0020_", "", regex=False)
        .str.replace("_x0020", "", regex=False)
        .str.strip()
    )

    # Convert date
    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"], dayfirst=True, errors="coerce"
    )

    # Rename price column
    df = df.rename(columns={"Modal_Price": "Price"})

    # Keep essential columns only
    df = df[
        [
            "Arrival_Date",
            "State",
            "Commodity",
            "Price"
        ]
    ]

    # Drop invalid rows
    df = df.dropna()

    # Date features
    df["Year"] = df["Arrival_Date"].dt.year
    df["Month"] = df["Arrival_Date"].dt.month
    df["Day"] = df["Arrival_Date"].dt.day
    df["DayOfWeek"] = df["Arrival_Date"].dt.dayofweek

    return df


def save_processed_data(df):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)


if __name__ == "__main__":
    df_raw = load_raw_data()
    df_processed = preprocess_data(df_raw)

    print("Rows after preprocessing:", len(df_processed))
    print(df_processed.head())
    print("\nColumns:")
    print(df_processed.columns)

    save_processed_data(df_processed)
    print("Preprocessing completed successfully.")

