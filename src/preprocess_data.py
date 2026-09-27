import pandas as pd

# Load the StockSmart bakery dataset
file_path = "data/Bakery sales.csv"

print("Loading StockSmart dataset...")

df = pd.read_csv(file_path)

# Remove the extra index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# Clean product names
df["article"] = df["article"].astype(str).str.strip().str.upper()

# Remove invalid product names
df = df[~df["article"].isin([".", "", "NAN"])]

# Remove rows missing important information
df = df.dropna(subset=["date", "article", "Quantity"])

# Remove duplicate rows
df = df.drop_duplicates()

# Aggregate transactions into daily product sales
daily_sales = (
    df.groupby(["date", "article"])["Quantity"]
    .sum()
    .reset_index()
)

# Sort chronologically
daily_sales = daily_sales.sort_values(["article", "date"])

print("\nOriginal transaction rows:", len(df))
print("Daily product records:", len(daily_sales))
print("\nFirst 10 daily sales records:")
print(daily_sales.head(10))

print("\nDate range:")
print(daily_sales["date"].min(), "to", daily_sales["date"].max())

print("\nNumber of products:")
print(daily_sales["article"].nunique())
# Save cleaned daily sales data
output_path = "data/cleaned_daily_sales.csv"
daily_sales.to_csv(output_path, index=False)

print("\nCleaned data saved to:", output_path)

print("\nStockSmart preprocessing completed successfully.")