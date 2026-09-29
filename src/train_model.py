import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
# 7-day historical mean baseline
baseline_predictions = X_test["rolling_7"]

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = mean_squared_error(y_test, baseline_predictions) ** 0.5
baseline_r2 = r2_score(y_test, baseline_predictions)

print("\n7-Day Historical Mean Baseline:")
print("MAE:", round(baseline_mae, 3))
print("RMSE:", round(baseline_rmse, 3))
print("R2:", round(baseline_r2, 3))
# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = mean_squared_error(y_test, linear_predictions) ** 0.5
linear_r2 = r2_score(y_test, linear_predictions)

print("\nLinear Regression:")
print("MAE:", round(linear_mae, 3))
print("RMSE:", round(linear_rmse, 3))
print("R2:", round(linear_r2, 3))
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
r2 = model.score(X_test, y_test)

print("\nModel evaluation:")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2:", round(r2, 3))

# Save trained model
os.makedirs("models", exist_ok=True)

model_path = "models/stocksmart_model.pkl"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
print("\nStockSmart model training completed successfully.")
