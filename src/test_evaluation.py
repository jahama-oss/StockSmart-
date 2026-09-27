"""
Test script for StockSmart model evaluation.

This uses sample bakery sales data to verify that the
evaluation functions are working correctly.
"""

from stocksmart_evaluation import (
    evaluate_model,
    compare_models,
    plot_predictions
)


# Sample actual bakery sales
actual_sales = [120, 135, 128, 150, 160, 155, 170]

# Example predictions from three forecasting methods
baseline_predictions = [118, 130, 132, 145, 158, 150, 165]

linear_predictions = [121, 133, 130, 148, 162, 154, 168]

random_forest_predictions = [120, 136, 127, 151, 159, 156, 169]


# Evaluate each model
baseline_results = evaluate_model(
    actual_sales,
    baseline_predictions,
    "7-Day Baseline"
)

linear_results = evaluate_model(
    actual_sales,
    linear_predictions,
    "Linear Regression"
)

random_forest_results = evaluate_model(
    actual_sales,
    random_forest_predictions,
    "Random Forest"
)


# Compare model performance
results = [
    baseline_results,
    linear_results,
    random_forest_results
]

comparison = compare_models(results)


# Display prediction graph
plot_predictions(
    actual_sales,
    random_forest_predictions,
    "Random Forest"
)
