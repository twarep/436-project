import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def create_model_df():
  # data column list
  column_list = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath',
                'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
                'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MoSold', 'SalePrice']


  raw_data_train = pd.read_csv('https://raw.githubusercontent.com/jmpark0808/pl_mnist_example/main/train_hp_msci436.csv')

  model_df = raw_data_train.select_dtypes(include = ['float64', 'int64']).fillna(0)
  model_df = model_df[column_list].astype(float)
  model_df = model_df.rename(columns= {'3SsnPorch':'ThreeSsnPorch'})
  return model_df

def train_model(): 
  model_df = create_model_df()
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
  return (weights, mse_test, r2, absolute_weighted_values, model_df)

def compute_min_and_range(arr):
  """
  Compute the minimum value and range (max - min) for each column of a NumPy ndarray.

  Returns:
      list: A list containing the [[min, range]] for each column of the input array.
  """
  result = []
  for column in arr.T:
      min_val = np.min(column)
      max_val = np.max(column)
      column_range = max_val - min_val
      result.append([min_val, column_range])
  return result

def calculate_min_range_values():
  # Calculating the predicted price
  model_df = create_model_df()
  return compute_min_and_range(model_df.values[:, 0:-1])


def number_input(keywords, minval, maxval):
   label = keywords
   test = st.number_input(label, step=1, min_value=minval, max_value=maxval)
   return (
     float(test)
   )

def standardize_user_inputs(model_df, user_inputs):
    """
    Standardizes user inputs based on the minimum value and range of each column.

    Parameters:
        min_max_values (list): Array of arrays containing the minimum value and range (max - min) for each column.
        user_inputs (list): List of user inputs to be standardized.

    Returns:
        list: A list of standardized user inputs.
    """

    standardized_inputs = []  # Initialize an empty list to store standardized inputs

    for i in range(len(user_inputs)):
        min_val, column_range = calculate_min_range_values()[i]  # Retrieve the minimum value and range for the current column

        # Apply standardization formula to each user input
        standardized_input = (user_inputs[i] - min_val) / column_range

        standardized_inputs.append(standardized_input)  # Append the standardized input to the list

    return standardized_inputs

def predict_home_price(LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, GrLivArea, BsmtFullBath,
            BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea,
              OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, MoSold):
  """
  Predict the price of a home given a set of parameters.

  Returns:
  price: Price of the user's home'.
  """
  weights, mse_test, r2, absolute_weighted_values, model_df = train_model()
  input = [LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, GrLivArea, BsmtFullBath,
            BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea,
              OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, MoSold]
  std_inputs = standardize_user_inputs(calculate_min_range_values, input)

  price = abs(sum([a*b for a,b in zip(weights,std_inputs)]))
  return price

def multiply_by_weights(LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, GrLivArea, BsmtFullBath,
                        BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea,
                        OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, MoSold):
  weights, mse_test, r2, absolute_weighted_values, model_df = train_model()
  input_variables = [LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, GrLivArea, BsmtFullBath,
                      BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea,
                      OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, MoSold]

  multiplied_values = []
  for input_var, weight in zip(input_variables, weights):
      multiplied_values.append(input_var * weight)

  return multiplied_values

