# StockSmart

StockSmart is a machine-learning bakery sales forecasting and production planning project developed for AIM 350 - Programming for AI.

## Project Goal

The goal of StockSmart is to predict daily bakery product sales using historical transaction data. Better sales forecasts can help small bakeries make more informed production decisions, reduce food waste, and decrease the risk of products selling out.

## Machine Learning Approach

This project uses supervised machine learning for regression.

The models evaluated include:

- 7-day historical mean baseline
- Linear Regression
- Random Forest Regressor

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Dataset

The project uses the French Bakery Daily Sales dataset.

The dataset contains approximately 234,000 bakery transaction records from January 2021 through September 2022.

Dataset source:
https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales

The raw dataset is not required to be stored directly in this repository. See the data-access instructions for information about obtaining the dataset.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Team Members

- Dylan Younghese
- Alen Lekic
- Alex Matjan
- Most Jahan

## Course

AIM 350 - Programming for AI  
Fall 2026
