import pandas as pd

# Load cleaned StockSmart sales data
file_path = "data/cleaned_daily_sales.csv"

print("Loading cleaned StockSmart data...")

df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"])

# Sort by product and date
df = df.sort_values(["article", "date"])

# Create calendar features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Create historical sales features for each product
df["lag_1"] = df.groupby("article")["Quantity"].shift(1)
df["lag_7"] = df.groupby("article")["Quantity"].shift(7)

df["rolling_7"] = (
    df.groupby("article")["Quantity"]
      .transform(lambda x: x.shift(1).rolling(7).mean())
)

# Remove rows that do not yet have enough historical data
features = df.dropna(
    subset=["lag_1", "lag_7", "rolling_7"]
).copy()

# Save feature dataset
output_path = "data/features.csv"
features.to_csv(output_path, index=False)

print("\nFeature engineering complete.")
print("Original daily records:", len(df))
print("Records ready for machine learning:", len(features))

print("\nFeatures created:")
print([
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_7"
])

print("\nFirst 10 feature records:")
print(features.head(10))

print("\nStockSmart feature dataset saved to:", output_path)