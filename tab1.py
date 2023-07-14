import streamlit as st
from model import multiply_by_weights, number_input, predict_home_price


# feature list
feature_list = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath',
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
               'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch', 'MoSold']

# months list and ordinal encoding
month_list = ['January', 'February', 'March', 'April',
              'May', 'June', 'July', 'August',
              'September', 'October', 'November', 'December']
month_encoding = {'January':1, 'February':2, 'March':3, 'April':4,
                  'May':5, 'June':6, 'July':7, 'August':8,
                  'September':9, 'October':10, 'November':11, 'December':12}

def tab1():
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
