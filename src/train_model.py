import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Load StockSmart feature dataset
file_path = "data/features.csv"

print("Loading StockSmart feature dataset...")

df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date")

# Features used by the model
feature_columns = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_7"
]

X = df[feature_columns]
y = df["Quantity"]

# Chronological 80/20 train-test split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining StockSmart model...")
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print("\nModel evaluation:")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))

# Save trained model
os.makedirs("models", exist_ok=True)

model_path = "models/stocksmart_model.pkl"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
print("\nStockSmart model training completed successfully.")