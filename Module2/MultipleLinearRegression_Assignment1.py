import pandas as pd
import numpy as np
import matplotlib.pyplot as plt







dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

train_data = pd.read_csv('../Data/kc_house_train_data.csv', dtype = dtype_dict)
test_data = pd.read_csv('../Data/kc_house_test_data.csv', dtype = dtype_dict)
