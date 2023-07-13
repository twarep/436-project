import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
















st.title('Sell with ML :house:')
tab1, tab2, tab3 = st.tabs(["Data Input", "Price Factors", "Recommendations and Visualizations"])



