
Project Title:

Predicting Next-Day Stock Log Returns Using Machine Learning and Recurrent Neural Networks

Author:
Waleed Alassaf

Overview
--------
This project predicts next-day stock log returns for four stocks:
AAPL, GOOGL, NVDA and TSLA.

The project compares 5 models ( Linear Regression, Ridge Regression, Random Forest, Recurrent Neural Network (RNN)) against Zero-return baseline


Files
-----
backend.py:	 Main Python script 

dashboard.py: 	 Streamlit dashboard used to display and compare the final model results.

AAPL.csv
GOOGL.csv
NVDA.csv
TSLA.csv
These csv files are historical market data used by the project

Requirements
------------
Python 3 is required with the following libraries.

1 pandas
2 numpy
3 yfinance
4 scikit-learn
5 tensorflow
6 streamlit
7 plotly


Running the Project
-------------------
1. Place the Python files and CSV files in the same folder.

2. Run the backend:

python backend.py

This executes the data preparation, feature engineering, model training
and evaluation process.

3. Run the Streamlit dashboard:

streamlit run dashboard.py


Notes
-----
1 The target variable is the next trading day's log return.
2 The data is kept in chronological ordering.
3 The main split uses the first 80% of the data for training and the final 20% for testing.
4 Evaluation includes RMSE, MAE, R-squared and directional accuracy.

