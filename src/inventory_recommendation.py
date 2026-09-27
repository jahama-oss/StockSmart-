import pandas as pd
import joblib
import math

print("Loading StockSmart inventory recommendation system...")

# Load feature data and trained model
df = pd.read_csv("data/features.csv")
model = joblib.load("models/stocksmart_model.pkl")

# Use the most recent record as an example
record = df.iloc[-1]

feature_columns = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_7"
]

X = pd.DataFrame([record[feature_columns]])

# Predict demand
predicted_demand = max(0, model.predict(X)[0])

# Example inventory settings
current_inventory = 10
safety_stock = 5
lead_time_days = 3

# Estimate inventory needed during supplier lead time
lead_time_demand = predicted_demand * lead_time_days

# Calculate reorder point
reorder_point = math.ceil(lead_time_demand + safety_stock)

# Determine whether an order should be placed
if current_inventory <= reorder_point:
    reorder_quantity = max(0, reorder_point - current_inventory)
    recommendation = "REORDER"
else:
    reorder_quantity = 0
    recommendation = "STOCK OK"

print("\n--- StockSmart Inventory Recommendation ---")
print("Product:", record["article"])
print("Date:", record["date"])
print("Predicted daily demand:", round(predicted_demand, 2))
print("Current inventory:", current_inventory)
print("Safety stock:", safety_stock)
print("Lead time:", lead_time_days, "days")
print("Reorder point:", reorder_point)
print("Recommendation:", recommendation)
print("Suggested order quantity:", reorder_quantity)

print("\nStockSmart inventory analysis completed successfully.")