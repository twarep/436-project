import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime







# data column list
column_list = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath',
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MoSold', 'SalePrice']

# feature list
feature_list = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath',
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
               'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch', 'MoSold']



# state initialization, each to 0
for feature in feature_list:
  if feature not in st.session_state:
    st.session_state[feature] = 0
if 'SalePrice' not in st.session_state:
  st.session_state['SalePrice'] = 0




# 80TH Percentile
model_80th_percentile = model_df[feature_list].quantile(0.8)

# months list and ordinal encoding
month_list = ['January', 'February', 'March', 'April',
              'May', 'June', 'July', 'August',
              'September', 'October', 'November', 'December']
month_encoding = {'January':1, 'February':2, 'March':3, 'April':4,
                  'May':5, 'June':6, 'July':7, 'August':8,
                  'September':9, 'October':10, 'November':11, 'December':12}

# Unstandardized input
x = model_df.values[:, 0:-1]

# Standardizing input
df_standardized = model_df.apply(lambda x: (x - x.min())/(x.max() - x.min()))
# Standardizing input params, but keeping the price unstandardized
X = df_standardized.values[:int(1.0*len(model_df.values)), 1:-1]
y = model_df.values[:, -1]

#creating a training and test split (80% for training and 20% for test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1066)

#training the regressor on the training data
reg = LinearRegression().fit(X_train, y_train)
weights = reg.coef_

X_val = model_df.values[int(0.8*len(model_df.values)):, 1:-1]

y_predict_std = reg.predict(X_test)
test_pred = reg.predict(X_val)

mse_test = mean_squared_error(test_pred, model_df.values[int(0.8*len(model_df.values)):, -1])
r2 = r2_score(y_true = y_test, y_pred = y_predict_std)

# Standardized weights
weights_total = np.sum(np.abs(weights))
absolute_weighted_values = weights/weights_total










st.title('Sell with ML :house:')
tab1, tab2, tab3 = st.tabs(["Data Input", "Price Factors", "Recommendations and Visualizations"])



