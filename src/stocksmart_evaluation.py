"""
StockSmart Model Evaluation
AIM 350 - Programming for AI

Author: Alex Matjan
Role: Model Evaluation & Data Visualization

This module evaluates StockSmart forecasting models using
MAE, RMSE, and R².
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(y_actual, y_predicted, model_name):
    """
    Calculate evaluation metrics for a StockSmart model.
    """

    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    r2 = r2_score(y_actual, y_predicted)

    results = {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return results


def compare_models(results):
    """
    Display model evaluation results in a table.
    """

    results_df = pd.DataFrame(results)

    print("\nStockSmart Model Evaluation")
    print("-" * 40)
    print(results_df.round(3))

    return results_df


def plot_predictions(y_actual, y_predicted, model_name):
    """
    Plot actual bakery sales against predicted sales.
    """

    plt.figure(figsize=(10, 5))

    plt.plot(y_actual, label="Actual Sales")
    plt.plot(y_predicted, label="Predicted Sales")

    plt.title(f"StockSmart - {model_name}")
    plt.xlabel("Day")
    plt.ylabel("Bakery Product Sales")
    plt.legend()
    plt.tight_layout()

    plt.show()
