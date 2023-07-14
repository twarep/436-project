import streamlit as st
from model import create_model_df
from tab1 import tab1
from tab2 import tab2
from tab3 import tab3

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

# show app tabs
st.title('Sell with ML :house:')
tab_1, tab_2, tab_3 = st.tabs(["Data Input", "Price Factors", "Recommendations and Visualizations"])
with tab_1:
  tab1()
with tab_2:
  tab2()
with tab_3:
  tab3()