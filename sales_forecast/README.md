# Sales Forecast Project

This repository contains a sales forecasting pipeline that compares baseline, machine learning, and time-series models for sales prediction. The goal is to evaluate several forecasting strategies on a retail sales dataset and compare their performance using metrics such as MAE, RMSE, WMAPE, and MAPE.

## Project Structure

- `load_data.py` - Loads the retail sales dataset from Hugging Face using streaming and samples rows for analysis.
- `clean_data.py` - Cleans the raw dataset, converts dates, sorts records, and creates time-based features.
- `metrics.py` - Defines evaluation metrics, including MAE, RMSE, WMAPE, and safe MAPE.
- `time_series_models.py` - Implements daily sales time-series forecasting methods:
  - Moving Average
  - Simple Exponential Smoothing
  - Holt-Winters
- `ml_models_with_lag_features.py` - Implements row-level sales forecasting using machine learning:
  - Linear Regression
  - XGBoost
  - XGBoost with lag and rolling-window features
- `baseline.py` - Placeholder for baseline or heuristic forecasting logic if needed.
- `results_comparison.py` - Builds a summary table of model performance and ranks models by task.
- `eda.py` - Exploratory data analysis utilities and plotting (not required for main runs).

## Dataset

The dataset is loaded from the Hugging Face dataset:

- `t4tiana/store-sales-time-series-forecasting`

Data is streamed, sampled, cleaned, and prepared before model training and evaluation.

## Dependencies

This project uses the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `statsmodels`
- `matplotlib`
- `datasets`

## Usage

Run the main scripts from the project root:

```bash
python load_data.py
python clean_data.py
python time_series_models.py
python ml_models_with_lag_features.py
python results_comparison.py
```

### Suggested workflow

1. Sample and inspect raw data with `load_data.py`.
2. Clean and enrich the data with `clean_data.py`.
3. Train and evaluate time-series forecasts with `time_series_models.py`.
4. Train and evaluate row-level machine learning forecasts with `ml_models_with_lag_features.py`.
5. Summarize and compare model performance with `results_comparison.py`.

## Model comparisons

The project currently compares two distinct forecasting tasks:

- Row-level sales prediction using ML models (`Linear Regression`, `XGBoost`, `XGBoost With Lag Features`).
- Daily total sales prediction using time-series models (`Moving Average`, `Simple Exponential Smoothing`, `Holt-Winters`).

`results_comparison.py` highlights the best model within each task using WMAPE and also ranks models by RMSE.

## Notes

- The row-level and daily total sales tasks are different, so their MAE/RMSE values should not be compared directly.
- WMAPE is often more appropriate for this sales forecasting problem than MAPE because it weights errors by actual sales volume.

## Improvements

Possible next steps for this project:

- Add a `requirements.txt` or `environment.yml` file for reproducibility.
- Add a dedicated `run_all.py` script to execute the full pipeline end to end.
- Add unit tests for preprocessing, forecasting, and metric calculations.
- Add more model tuning and cross-validation for the machine learning pipeline.
