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


raw_data_train = pd.read_csv('https://raw.githubusercontent.com/jmpark0808/pl_mnist_example/main/train_hp_msci436.csv')

model_df = raw_data_train.select_dtypes(include = ['float64', 'int64']).fillna(0)
model_df = model_df[column_list].astype(float)
model_df = model_df.rename(columns= {'3SsnPorch':'ThreeSsnPorch'})

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
# Calculating the predicted price
min_range_values = compute_min_and_range(model_df.values[:, 0:-1])


def number_input(keywords, minval, maxval):
   label = keywords
   test = st.number_input(label, step=1, min_value=minval, max_value=maxval)
   return (
     float(test)
   )

def standardize_user_inputs(min_range_values, user_inputs):
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
        min_val, column_range = min_range_values[i]  # Retrieve the minimum value and range for the current column

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
  input = [LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, GrLivArea, BsmtFullBath,
            BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea,
              OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, MoSold]
  std_inputs = standardize_user_inputs(min_range_values, input)

  price = abs(sum([a*b for a,b in zip(weights,std_inputs)]))
  return price

def multiply_by_weights(LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, GrLivArea, BsmtFullBath,
                        BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea,
                        OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, MoSold):
  input_variables = [LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, GrLivArea, BsmtFullBath,
                      BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea,
                      OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, MoSold]

  multiplied_values = []
  for input_var, weight in zip(input_variables, weights):
      multiplied_values.append(input_var * weight)

  return multiplied_values

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


top_three = sorted(range(len(weights)), key=lambda i: weights[i])[-3:] #getting the top 3 factors based on weight

#function prints the top three features based on weight in linear regression

def top_factors(top, facts):
  st.header("The top 3 factors helping increase your home price are:")

  for a in range(len(top)):
    notation = str(a+1) + ". " + facts[top[a]]
    st.write(notation)


st.title('Sell with ML :house:')
tab1, tab2, tab3 = st.tabs(["Data Input", "Price Factors", "Recommendations and Visualizations"])
#---

with tab1:
  st.write("We use a Decision Support System (DSS) and linear regression model with the MNIST House Price data set to effectively analyze large amounts of data and predict house prices. This allows us to identify significant factors affecting house prices and perform a sensitivity analysis, enhancing our understanding and providing further insights through visualized results.")
  st.header("Input the characteristics of your house below:")
  st.subheader("Property Measurements (square feet)")
  LotArea = number_input("Lot Size:",1000,250000)
  GrLivArea = number_input("Above Ground Living Area:",1000,10000)
  st.subheader("Number of Rooms Above Ground", help="Total number of rooms includes extra rooms besides the bedrooms,\
                kitchens, and bathrooms. Dining rooms, living rooms, office rooms, and others fit under this category.")
  TotRmsAbvGrd = number_input("Total Number of Rooms: (includes extra rooms on top of the ones below)",0,20)
  BedroomAbvGr = number_input("Bedrooms:",0,10)
  KitchenAbvGr = number_input("Kitchens:",0,5)
  FullBath = number_input("Full Bathrooms:",0,5)
  HalfBath = number_input("Half Bathrooms:",0,5)
  st.subheader("When was your home built?")
  YearBuilt = number_input("Original year of construction:",1900,2020)
  YearRemodAdd = number_input("Year Remodeled (same as construction date if no remodeling or additions):",1900,2020)
  with st.expander("I have a Basement:"):
    TotalBsmtSF = number_input("Basement Area:",0,10000)
    BsmtFullBath = number_input("Basement Full Bathrooms:",0,5)
    BsmtHalfBath = number_input("Basement Half Bathrooms:",0,5)
  with st.expander("I have a Garage:"):
    GarageCars = number_input("Car Capacity:",0,10)
    GarageArea = number_input("Garage Area (in square feet):",0,5000)
  with st.expander("I have a Porch:"):
    st.subheader("Square Feet of Each Type of Porch")
    OpenPorchSF = number_input("Open Porch:",0,1000)
    EnclosedPorch = number_input("Enclosed Porch:",0,1000)
    ThreeSsnPorch = number_input("Three Season Porch:",0,1000)
    ScreenPorch = number_input("Screened Porch:",0,1000)
  st.subheader("How Would You Rate the Quality of Your Home?")
  OverallQual = st.slider('Rate the overall material and finish of the house:', 1, 10, 5)
  OverallCond = st.slider('Rate the overall condition of the house:', 1, 10, 5)
  # MoSold = OverallQual = st.slider('When are you selling the house?', 0, 12, 5)
  ChosenMonth = st.selectbox('What month are you planning on selling the house?', month_list)
  MoSold = month_encoding[ChosenMonth]

  # update states
  for feature in feature_list:
    st.session_state[feature] = locals()[feature]

  if st.button('Predict the price'):
    price = predict_home_price(LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF,
  GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath,
  BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea, MoSold,
  OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch)
    st.session_state["SalePrice"] = price
    st.write(price)

    st.divider()
    st.write('Use the links above to head to the second page and learn how different factors affect the price of your home!')

    weighted_arr = multiply_by_weights(LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF,
    GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath,
    BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea, MoSold,
    OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch)
      # st.write(weighted_arr)
      # st.header('weights')
      # st.write(weights)

with tab2:
  st.header('How to increase the price of :blue[your house] :house:')
  current_price = "$" + str(st.session_state.SalePrice)
  user_price_string = "Your current selling price is: " + current_price
  st.subheader(user_price_string)
  top_factors(top_three, feature_list) #calling function to print top three factors
  st.write("We recommend you consider renovating the house and sprucing up the place with fresh paint to improve the overall condition of the house!")
  st.write("We recommend you consider adding additional rooms to the house. Many buyers zero in on the rooms as the central feature of a home, so if you don't have many rooms, it can ultimately affect how much you garner from a sale.")
  st.write("Break up two-story walls and expand the amount of space in your garage!")



with tab3:
  st.subheader("Visualizations of Variables:")
  labels = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
  'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
  'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'MoSold',
  'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch']

  values = absolute_weighted_values

  # Use `hole` to create a donut-like pie chart
  fig_target = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
  st.plotly_chart(fig_target, use_container_width=True)

  st.subheader("Correlation Between Features and Price")
  st.write("""Using a heatmap, we can evaluate the features that have a higher (darker) correlation to the housing price.
  From this visualization, the overall house quality has the highest correlation.""")

  heatmap = px.imshow(model_df.corr(), text_auto=True)
  st.plotly_chart(heatmap, use_container_width=True)
  st.subheader("Model Accuracy")
  st.write("Using the R score, we can compare the regular model's accuracy vs the standardized model's accuracy")


  col1, col2 = st.columns(2)
  col1.metric("R-score", r2, "")
  # MSE was computed in the computations above:
  col2.metric("MSE Test Score", "39440790871.19446", "")



