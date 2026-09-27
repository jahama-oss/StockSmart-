import pandas as pd
import joblib

# File locations
model_path = "models/stocksmart_model.pkl"
data_path = "data/features.csv"

print("Loading StockSmart prediction system...")

# Load trained model
model = joblib.load(model_path)

# Load feature data
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])

# These must match the features used during training
feature_columns = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_7"
]

# Use the most recent record as a test prediction
sample = df.iloc[-1]

X_new = sample[feature_columns].to_frame().T

prediction = model.predict(X_new)[0]

print("\nProduct:", sample["article"])
print("Date:", sample["date"].date())
print("Actual quantity:", sample["Quantity"])
print("Predicted quantity:", round(prediction, 2))

difference = abs(sample["Quantity"] - prediction)

print("Prediction error:", round(difference, 2))

print("\nStockSmart prediction completed successfully.")