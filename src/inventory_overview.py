import pandas as pd
import joblib
import math

print("Loading StockSmart inventory overview...")

# Load feature data and trained model
df = pd.read_csv("data/features.csv")
model = joblib.load("models/stocksmart_model.pkl")

# Features used by the ML model
feature_columns = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_7"
]

# Inventory settings
current_inventory = 10
safety_stock = 5
lead_time = 3

results = []

# Analyze each product
for product in sorted(df["article"].unique()):

    product_data = df[df["article"] == product].sort_values("date")

    if product_data.empty:
        continue

    latest = product_data.iloc[-1]

    X = latest[feature_columns].to_frame().T

    predicted_demand = max(0, float(model.predict(X)[0]))

    reorder_point = math.ceil(
        predicted_demand * lead_time + safety_stock
    )

    suggested_order = max(
        0,
        reorder_point - current_inventory
    )

    if current_inventory <= reorder_point:
        status = "REORDER"
    else:
        status = "OK"

    results.append({
        "Product": product,
        "Predicted Daily Demand": round(predicted_demand, 2),
        "Current Inventory": current_inventory,
        "Reorder Point": reorder_point,
        "Suggested Order": suggested_order,
        "Status": status
    })

# Create inventory overview table
overview = pd.DataFrame(results)

# Put products needing reorder first
overview = overview.sort_values(
    ["Status", "Predicted Daily Demand"],
    ascending=[False, False]
)

print("\n=== StockSmart Inventory Overview ===\n")
print(overview.to_string(index=False))

print("\nProducts analyzed:", len(overview))
print(
    "Products requiring reorder:",
    (overview["Status"] == "REORDER").sum()
)

# Save report
output_path = "data/inventory_overview.csv"
overview.to_csv(output_path, index=False)

print("\nInventory overview saved to:", output_path)
print("\nStockSmart inventory overview completed successfully.")