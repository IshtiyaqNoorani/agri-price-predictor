import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = Path("data/processed/weekly_data.csv")
MODEL_DIR = Path("models/weekly_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("Total weekly rows:", len(df))
print("Total commodities:", df["Commodity"].nunique())

# --------------------------------------------------
# Train one model per commodity
# --------------------------------------------------
results = []

for commodity in sorted(df["Commodity"].unique()):
    sub_df = df[df["Commodity"] == commodity]

    # Skip commodities with too little data
    if len(sub_df) < 30:
        continue

    X = sub_df.drop(columns=["Price", "Commodity"])
    y = sub_df["Price"]

    categorical_features = ["State"]
    numerical_features = ["Year", "Month", "DayOfWeek"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    model_path = MODEL_DIR / f"{commodity.replace(' ', '_').lower()}.pkl"
    joblib.dump(pipeline, model_path)

    results.append((commodity, len(sub_df), mae))

    print(f"{commodity:30s} | rows: {len(sub_df):4d} | MAE: {mae:8.2f}")

# --------------------------------------------------
# Summary
# --------------------------------------------------
print("\n=== SUMMARY ===")
for commodity, rows, mae in sorted(results, key=lambda x: x[2]):
    print(f"{commodity:30s} | rows: {rows:4d} | MAE: {mae:8.2f}")

