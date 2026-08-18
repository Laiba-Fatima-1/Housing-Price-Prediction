"""
Task 3: California Housing Price Prediction
Trains a Random Forest Regressor on the California Housing dataset.
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 1. Load ----------
df = pd.read_csv("housing.csv")
print(f"Loaded {len(df)} records")
print(df.isnull().sum())

# ---------- 2. Clean ----------
# total_bedrooms has some missing values -> fill with median
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# ---------- 3. Feature engineering ----------
# Per-household ratios are far more predictive than raw totals
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# One-hot encode the categorical location feature
df = pd.get_dummies(df, columns=["ocean_proximity"], prefix="ocean")

feature_cols = [c for c in df.columns if c != "median_house_value"]
X = df[feature_cols]
y = df["median_house_value"]

# ---------- 4. Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 5. Train ----------
model = RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ---------- 6. Evaluate ----------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nMAE:  ${mae:,.0f}")
print(f"RMSE: ${rmse:,.0f}")
print(f"R2:   {r2:.4f}")

# ---------- 7. Feature importance ----------
importances = sorted(
    zip(feature_cols, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
print("\nTop features:")
for name, imp in importances[:6]:
    print(f"  {name}: {imp:.3f}")

# ---------- 8. Save ----------
joblib.dump(model, "housing_model.joblib")
joblib.dump(feature_cols, "feature_cols.joblib")

metrics = {
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "train_size": len(X_train),
    "test_size": len(X_test),
    "feature_importances": [[n, float(i)] for n, i in importances],
    "price_min": float(y.min()),
    "price_max": float(y.max()),
    "price_mean": float(y.mean()),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: housing_model.joblib, feature_cols.joblib, metrics.json")
