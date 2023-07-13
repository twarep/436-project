
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

