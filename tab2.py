import streamlit as st
from model import train_model

# feature list
feature_list = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath',
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
               'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch', 'MoSold']

weights, mse_test, r2, absolute_weighted_values, model_df = train_model()

def tab2():
  def top_factors(top, facts):
  #function prints the top three features based on weight in linear regression
    st.header("The top 3 factors helping increase your home price are:")
    for a in range(len(top)):
      notation = str(a+1) + ". " + facts[top[a]]
      st.write(notation)

  top_three = sorted(range(len(weights)), key=lambda i: weights[i])[-3:] #getting the top 3 factors based on weight
  st.header('How to increase the price of :blue[your house] :house:')
  current_price = "$" + str(st.session_state.SalePrice)
  user_price_string = "Your current selling price is: " + current_price
  st.subheader(user_price_string)
  top_factors(top_three, feature_list) #calling function to print top three factors
  st.write("We recommend you consider renovating the house and sprucing up the place with fresh paint to improve the overall condition of the house!")
  st.write("We recommend you consider adding additional rooms to the house. Many buyers zero in on the rooms as the central feature of a home, so if you don't have many rooms, it can ultimately affect how much you garner from a sale.")
  st.write("Break up two-story walls and expand the amount of space in your garage!")
