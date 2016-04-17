import numpy as np

def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1 # add a constant column to the data frame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_frame given by the â??featuresâ?? list into the panda â??features_sframeâ??
    features_matrix = data_frame.as_matrix(features)
    # assign the column of data_frame associated with the target to the variable â??output_sarrayâ??
    output_array = data_frame.as_matrix(output) 
    return(features_matrix, output_array)

def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)

