# StockSmart

StockSmart is a machine-learning bakery sales forecasting and production planning project developed for AIM 350 - Programming for AI.

## Project Goal

The goal of StockSmart is to predict daily bakery product sales using historical transaction data. Better sales forecasts can help small bakeries make more informed production decisions, reduce food waste, and decrease the risk of products selling out.

## Machine Learning Approach

This project uses supervised machine learning for regression.

The models evaluated include:

* 7-day historical mean baseline
* Linear Regression
* Random Forest Regressor

Model performance is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

## Dataset

The project uses the French Bakery Daily Sales dataset.

The dataset contains approximately 234,000 bakery transaction records from January 2021 through September 2022.

Dataset source:
https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales

The raw dataset is not required to be stored directly in this repository. See the data-access instructions for information about obtaining the dataset.

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Jupyter Notebook

## Team Members

* Dylan Younghese
* Alen Lekic
* Alex Matjan
* Most Jahan

## Course

AIM 350 - Programming for AI  
Fall 2026



\## Project Structure



StockSmart/

\- app.py - Streamlit inventory dashboard

\- requirements.txt - Python package requirements

\- src/

&#x20; - preprocess\_data.py - Cleans and prepares the bakery sales data

&#x20; - create\_features.py - Creates features used for machine learning

&#x20; - train\_model.py - Trains and evaluates the forecasting models

&#x20; - predict.py - Generates sales predictions

&#x20; - inventory\_recommendation.py - Creates inventory recommendations

&#x20; - inventory\_overview.py - Generates the multi-product inventory overview

\- data/ - Contains the bakery dataset and generated data files

\- models/ - Contains generated trained model files



\## Setup Instructions



1\. Clone or download the StockSmart repository.



2\. Open PowerShell or a terminal and navigate to the StockSmart project folder.



3\. Install the required Python packages:



&#x20;   pip install -r requirements.txt



4\. Download the French Bakery Daily Sales dataset from Kaggle.



5\. Place the bakery sales CSV file in the `data` folder.



\## Run the Project



Run the project scripts in this order:



&#x20;   python src/preprocess\_data.py

&#x20;   python src/create\_features.py

&#x20;   python src/train\_model.py

&#x20;   python src/predict.py

&#x20;   python src/inventory\_recommendation.py

&#x20;   python src/inventory\_overview.py



After the scripts finish successfully, launch the StockSmart dashboard:



&#x20;   streamlit run app.py



The dashboard will open in your web browser.



\## Generated Files



Some generated CSV files and the `models/` folder are excluded from GitHub using `.gitignore`. These files are created automatically when the project scripts are run in the correct order.



\## Dashboard



The StockSmart dashboard allows users to:

\- Select bakery products

\- View predicted daily demand

\- Set current inventory, safety stock, and supplier lead time

\- Calculate reorder points

\- Receive reorder recommendations

\- View product sales history

\- Monitor inventory across all products

