import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from model import train_model

weights, mse_test, r2, absolute_weighted_values, model_df = train_model()

def tab3():
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
