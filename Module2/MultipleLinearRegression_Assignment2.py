import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

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
    predictions = np.dot(feature_matrix,weights)
    return(predictions)

def feature_derivative(errors, feature):
    derivative = 2*np.dot(errors, feature)
    derivative /= np.size(errors)
    return(derivative)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        errors = np.subtract(predictions, np.transpose(output))
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:,i])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += derivative
            # update the weight based on step size and derivative:
            weights[i] -= step_size*derivative
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

test_data = pd.read_csv('../Data/kc_house_test_data.csv', dtype = dtype_dict)
train_data = pd.read_csv('../Data/kc_house_train_data.csv', dtype = dtype_dict)

(feature_matrix, output) = get_numpy_data(train_data, ['sqft_living'], ['price'])

initial_weights = np.array([-47000.,1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)

print('sol',simple_weights)

(test_feature_matrix, test_output) = get_numpy_data(test_data, ['sqft_living'], ['price'])

test_predictions = predict_outcome(test_feature_matrix, simple_weights)

print('test prediciton', test_predictions)

errors = np.subtract(test_predictions, test_output[:,0])
rss = np.dot(errors,errors)

print('rss', rss)

model_features = ['sqft_living', 'sqft_living15']
my_output = ['price']
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)

print('sol2', weights)

(test_feature_matrix, test_output) = get_numpy_data(test_data, ['sqft_living', 'sqft_living15'], ['price'])
test_predictions = predict_outcome(test_feature_matrix, weights)

print('test prediciton', test_predictions)

errors = np.subtract(test_predictions, test_output[:,0])
rss2 = np.dot(errors,errors)

print('rss', rss2)

