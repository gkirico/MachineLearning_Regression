import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1 # add a constant column to the data frame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_frame given by the ‘features’ list into the panda ‘features_sframe’
    features_matrix = data_frame.as_matrix(features)
    # assign the column of data_frame associated with the target to the variable ‘output_sarray’
    output_array = data_frame.as_matrix(output) 
    return(features_matrix, output_array)

def predict_outcome(feature_matrix, weights):
    pass
    return(predictions)

def feature_derivative(errors, feature):
    pass
    return(derivative)

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

train_data = pd.read_csv('../Data/kc_house_data.csv', dtype = dtype_dict)

(features_matrix, output_array) = get_numpy_data(train_data, ['sqft_living','bedrooms'], ['price'])
