import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = Path("data/processed/weekly_data.csv")
MODEL_PATH = Path("models/price_model_weekly.pkl")

# --------------------------------------------------
# Load data
# --------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError("weekly_data.csv not found")

df = pd.read_csv(DATA_PATH)

print("Weekly rows:", len(df))

# --------------------------------------------------
# Features & target
# --------------------------------------------------
X = df.drop(columns=["Price"])
y = df["Price"]

categorical_features = ["Commodity", "State"]
numerical_features = ["Year", "Month", "DayOfWeek"]

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
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

# --------------------------------------------------
# Train / test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
preds = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Weekly MAE: {mae:.2f}")
print(f"Weekly R²: {r2:.3f}")

# --------------------------------------------------
# Save model
# --------------------------------------------------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"Weekly model saved to {MODEL_PATH}")

