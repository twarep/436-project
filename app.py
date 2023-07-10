import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

raw_data_train = pd.read_csv('https://raw.githubusercontent.com/jmpark0808/pl_mnist_example/main/train_hp_msci436.csv')

# select only the numerical data types
df_num = raw_data_train.select_dtypes(include = ['float64', 'int64'])
df_num.head()

model_df = raw_data_train.select_dtypes(include = ['float64', 'int64']).fillna(0)
model_df = model_df[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath',
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MoSold', 'SalePrice']]
model_df = model_df.rename(columns= {'3SsnPorch':'ThreeSsnPorch'})
model_df.info()

model_df.tail()
X = model_df.values[:, 0:-1]
y = model_df.values[:, -1]

reg = LinearRegression().fit(X, y)
weights = reg.coef_
weights = weights.tolist()

# Cheatsheet available at https://docs.streamlit.io/library/cheatsheet

raw_data_train = pd.read_csv('https://raw.githubusercontent.com/jmpark0808/pl_mnist_example/main/train_hp_msci436.csv')


model_df = raw_data_train.select_dtypes(include = ['float64', 'int64']).fillna(0)
model_df = model_df[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath',
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MoSold', 'SalePrice']].astype(float)
model_df = model_df.rename(columns= {'3SsnPorch':'ThreeSsnPorch'})



X = model_df.values[:, 0:-1]
y = model_df.values[:, -1]
reg = LinearRegression().fit(X, y)
weights = reg.coef_
weights = weights.tolist()




def predict_home_price(LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, GrLivArea, BsmtFullBath,
            BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea,
              OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, MoSold):
  input=[LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, GrLivArea, BsmtFullBath,
            BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea,
              OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, MoSold]
  price = sum([a*b for a,b in zip(weights,input)])
  return price

def number_input(keywords, minval, maxval):
## TODO: discuss what is max values
   label = keywords
   test = st.number_input(label, step=1, min_value=minval, max_value=maxval )
   return (
     float(test)
   )

st.header('DSS Hello App Title Here')

#---

LotArea = number_input("Enter the lot size in square feet",0,100000)
OverallQual = st.slider('Rate the overall material and finish of the house', 0, 10, 5)
OverallCond = st.slider('Rate the overall condition of the house', 0, 10, 5)
YearBuilt = number_input("Original construction date",0,100000)
YearRemodAdd = number_input("Remodel date (same as construction date if no remodeling or additions)",0,100000)
TotalBsmtSF = number_input("Total square feet of basement area",0,100000)
GrLivArea = number_input("Above ground living area square feet",0,100000)
BsmtFullBath = number_input("Basement full bathrooms",0,100000)
BsmtHalfBath = number_input("Basement half bathrooms",0,100000)
FullBath = number_input("Full bathrooms above ground",0,100000)
HalfBath = number_input("Half bathrooms above ground",0,100000)
BedroomAbvGr = number_input("Bedrooms above ground (does NOT include basement bedrooms)",0,100000)
KitchenAbvGr = number_input("Kitchens above ground (does NOT include basement kitchens)",0,100000)
TotRmsAbvGrd = number_input("TotRmsAbvGrd",0,100000)
GarageCars = number_input("Size of garage in car capacity",0,10)
GarageArea = number_input("Size of garage in square feet",0,100000)
MoSold = OverallQual = st.slider('When are you selling the house?', 0, 12, 5)


OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch = (0,0,0,0)

porch = st.radio(
    "Which porch would you like?", ("OpenPorchSF", "EnclosedPorch", "ThreeSsnPorch", "ScreenPorch"))

## kind of hacky but it should work?
if porch == "OpenPorchSF":
  OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch =[1,0,0,0]
elif porch == "EnclosedPorch":
  OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch =[0,1,0,0]
elif porch == "ThreeSsnPorch":
  OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch =[0,0,1,0]
else:
  OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch =[0,0,0,1]


if st.button('Predict the price'):
  price = predict_home_price(LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF,
GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath,
BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, GarageCars, GarageArea, MoSold,
OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch)
  st.write(weights)
  st.write(price)




labels = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'MoSold',
'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch']

values = [4500, 2500, 1053, 500]

# Use `hole` to create a donut-like pie chart
fig_target = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
st.plotly_chart(fig_target, use_container_width=True)

