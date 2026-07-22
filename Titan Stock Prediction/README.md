# Titan Stock Price Prediction Using Machine Learning

## Project Overview

The project applies machine learning techniques to predict the next day's stock price movement of Titan Company Ltd. using historical stock market data obtained from Yahoo Finance. The project follows a complete machine learning workflow, including data collection, exploratory data analysis, feature engineering, model training, performance evaluation, and comparison of multiple classification models.

The objective of this project is to determine whether the stock price will increase (1) or decrease/remain unchanged (0) on the next trading day based on historical market information and engineered financial features.

## Objectives

- Collect historical stock price data for Titan Company Ltd.
- Perform exploratory data analysis to understand the dataset.
- Engineer meaningful financial features from raw market data.
- Train multiple machine learning classification models.
- Evaluate and compare model performance.
- Identify the most influential features affecting stock movement.

## Dataset

Company: Titan Company Ltd.
Ticker Symbol: TITAN.NS
Data Source: Yahoo Finance (yfinance)
Time Period: Last 10 Years

## Original Features

- Date
- Open
- High
- Low
- Close
- Volume
- Dividends
- Stock Splits

The raw dataset is stored in:

data/raw/titan_raw.csv

# Project Workflow

## 1. Data Collection

Historical stock market data was downloaded using the 'yfinance' library.

Script used:
src/download_data.py

The downloaded dataset was saved as:
data/raw/titan_raw.csv

## 2. Exploratory Data Analysis (EDA)

The dataset was explored to understand its structure and characteristics.
The following analyses were performed:
- Dataset overview
- Data types
- Descriptive statistics
- Missing value analysis

Scripts used:
src/explore_data.py

## 3. Data Visualization

Several visualizations were created to better understand historical stock behaviour. Shows the relation of closing price and moving averages in those ten years.

Scripts used:
src/visualize_data.py

The generated figures are stored in:
plots/

## 4. Feature Engineering

To improve predictive performance, several financial features were created from the original dataset.
Engineered features include:

- Daily Return
- 5-Day Moving Average (MA5)
- 20-Day Moving Average (MA20)
- MA5 / MA20 Ratio
- Price / MA20 Ratio
- Lag-1 Return
- Lag-2 Return
- 5-Day Rolling Volatility
- Volume Change

### Target Variable

The prediction target is defined as:

- 1 → Next day's closing price is higher than today's closing price.
- 0 → Next day's closing price is lower than or equal to today's closing price.

The processed dataset is saved as:
data/processed/titan_features.csv

Script used:
src/feature_engineering.py

## 5. Machine Learning Models

Four classification models were trained and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
The models were trained using an 80:20 train-test split.

Script used:
src/train_model.py

## 6. Model Evaluation

The trained models were evaluated using the following metrics:

- Training Accuracy
- Testing Accuracy
- Confusion Matrix
- Classification Report
- Feature Importance (XGBoost)
- Model Comparison

To better understand model performance, an underfitting vs overfitting analysis was also performed using Decision Trees with different maximum depths and its plot is stored in:
plots/

# Project Structure

TITAN STOCK PREDICTION
Contains four major folders with README.md ==>
1. data: which includes raw and processed data as titan_raw.csv and titan_features.csv respectively.
2. plots: Contains 5 plots => model_comparison.png, moving_average.png, underfitting_vs_overfitting.png, xgb_confusion_matrix.png, xgb_feature_importance.png
3. Results: classification_report.txt, feature_importance.csv, project_summary.txt
4. src: download_data.py, explore_data.py, feature_engineering.py, train_model.py, visualize_data.py

# Output Files

## Plots
The following visualizations are generated during model training:

- Model Comparison
- Moving Average
- Underfitting vs Overfitting Analysis
- XGBoost Confusion Matrix
- XGBoost Feature Importance
These are stored inside the plots/

## Results
The results/ contains:

- Classification Report
- Feature Importance
- Project Summary

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- yfinance

# How to Run the Project

## Execute the scripts in the following order
download_data.py ==> explore_data.py ==> visualize_data.py ==> feature_engineering.py ==> train_model.py

# Future Improvements

Possible improvements to the project include:

- Hyperparameter tuning
- Time-series cross-validation
- Real-time stock price prediction

# Project by:
Vaidehi Ghadmode
25B1820
