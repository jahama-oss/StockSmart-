import streamlit as st
import pandas as pd
import joblib
import math

# Page setup
st.set_page_config(
    page_title="StockSmart",
    page_icon="📦",
    layout="wide"
)

st.title("📦 StockSmart")
st.subheader("AI-Powered Inventory Management Dashboard")

# Load data and trained model
@st.cache_data
def load_data():
    df = pd.read_csv("data/features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/stocksmart_model.pkl")

df = load_data()
model = load_model()

# Product selector
products = sorted(df["article"].unique())

selected_product = st.selectbox(
    "Select Product",
    products
)

product_data = df[df["article"] == selected_product].copy()
record = product_data.iloc[-1]

# Features used by machine learning model
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

predicted_demand = max(0, model.predict(X)[0])

# Inventory controls
st.sidebar.header("Inventory Settings")

current_inventory = st.sidebar.number_input(
    "Current Inventory",
    min_value=0,
    value=10
)

safety_stock = st.sidebar.number_input(
    "Safety Stock",
    min_value=0,
    value=5
)

lead_time_days = st.sidebar.number_input(
    "Supplier Lead Time (Days)",
    min_value=1,
    value=3
)

# Inventory calculations
lead_time_demand = predicted_demand * lead_time_days
reorder_point = math.ceil(lead_time_demand + safety_stock)

if current_inventory <= reorder_point:
    recommendation = "REORDER"
    reorder_quantity = max(
        0,
        reorder_point - current_inventory
    )
else:
    recommendation = "STOCK OK"
    reorder_quantity = 0

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predicted Daily Demand",
    f"{predicted_demand:.2f}"
)

col2.metric(
    "Current Inventory",
    current_inventory
)

col3.metric(
    "Reorder Point",
    reorder_point
)

col4.metric(
    "Suggested Order",
    reorder_quantity
)

st.divider()

# Recommendation
st.subheader("Inventory Recommendation")

if recommendation == "REORDER":
    st.error(
        f"⚠️ REORDER {reorder_quantity} units of {selected_product}"
    )
else:
    st.success(
        f"✅ Stock level for {selected_product} is OK"
    )

# Sales history
st.subheader("Product Sales History")

chart_data = (
    product_data
    .set_index("date")[["Quantity"]]
)

st.line_chart(chart_data)

# Recent data
st.subheader("Recent Product Data")

st.dataframe(
    product_data.tail(10),
    use_container_width=True
)

st.caption(
    "StockSmart uses machine learning and historical sales data "
    "to support inventory planning and reorder decisions."

)
# Multi-product inventory overview
st.divider()
st.subheader("Inventory Overview")

try:
    inventory_overview = pd.read_csv("data/inventory_overview.csv")

    total_products = len(inventory_overview)
    reorder_products = (
        inventory_overview["Status"] == "REORDER"
    ).sum()
    ok_products = (
        inventory_overview["Status"] == "OK"
    ).sum()

    overview_col1, overview_col2, overview_col3 = st.columns(3)

    overview_col1.metric(
        "Products Monitored",
        total_products
    )

    overview_col2.metric(
        "Need Reorder",
        reorder_products
    )

    overview_col3.metric(
        "Stock OK",
        ok_products
    )

    st.subheader("Products Requiring Attention")

    reorder_table = inventory_overview[
        inventory_overview["Status"] == "REORDER"
    ]

    st.dataframe(
        reorder_table,
        width="stretch"
    )

except FileNotFoundError:
    st.warning(
        "Inventory overview has not been generated yet."
    )

