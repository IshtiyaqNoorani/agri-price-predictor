import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/processed/cleaned_data.csv")
MODEL_PATH = Path("models/price_model.pkl")


# -----------------------------
# Load data
# -----------------------------
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "cleaned_data.csv not found. Run preprocess.py first."
        )
    return pd.read_csv(DATA_PATH)


# -----------------------------
# Train model
# -----------------------------
def train_model(df: pd.DataFrame):
    # Features & target
    X = df.drop(columns=["Price", "Arrival_Date"])
    y = df["Price"]

    # Categorical & numerical columns
    categorical_features = ["Commodity", "State"]
    numerical_features = [
        "Year", "Month", "Day", "DayOfWeek"
    ]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")

    return pipeline


# -----------------------------
# Save model
# -----------------------------
def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# -----------------------------
# Run training
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    save_model(model)

