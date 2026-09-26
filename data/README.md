# StockSmart Dataset

## Dataset

StockSmart uses the French Bakery Daily Sales dataset.

The dataset contains approximately 234,000 bakery transaction records collected from January 2021 through September 2022.

The main fields include:
Date
Time
Ticket number
Article/product name
Quantity
Unit price

## Dataset Source

French Bakery Daily Sales:
https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales

## Data Preprocessing

For the StockSmart project, the transaction data is cleaned and aggregated into daily product sales.

The preprocessing process includes:
Parsing dates into datetime format
Checking for missing values
Checking for duplicate records
Standardizing product names
Aggregating quantity sold by date and product
Creating calendar features such as day of week, month, and weekend status
Creating historical sales features such as lagged sales and rolling averages
Sorting observations chronologically for model training and testing

The raw dataset is not stored directly in this repository. It can be downloaded from the dataset source above.
