import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

train_data = pd.read_csv('../Data/kc_house_train_data.csv', dtype = dtype_dict)
test_data = pd.read_csv('../Data/kc_house_test_data.csv', dtype = dtype_dict)

test_data['bedrooms_squared'] = test_data['bedrooms']*test_data['bedrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']
test_data['log_sqft_living'] = np.log(test_data['sqft_living'])
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

print('test avg sq beds', np.mean(test_data['bedrooms_squared']))
print('test avg bed bath', np.mean(test_data['bed_bath_rooms']))
print('test avg log sqft', np.mean(test_data['log_sqft_living']))
print('test avg lat long', np.mean(test_data['lat_plus_long']))

train_data['bedrooms_squared'] = train_data['bedrooms']*train_data['bedrooms']
train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
train_data['log_sqft_living'] = np.log(train_data['sqft_living'])
train_data['lat_plus_long'] = train_data['lat'] + train_data['long']

model1 = pd.stats.api.ols(y=train_data['price'], x=train_data[['sqft_living','bedrooms', 'bathrooms', 'lat', 'long']])
model2 = pd.stats.api.ols(y=train_data['price'], x=train_data[['sqft_living','bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']])
model3 = pd.stats.api.ols(y=train_data['price'], x=train_data[['sqft_living','bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']])

print('model 1', model1)
print('model 2', model2)
print('model 3', model3)

print('rss1', np.dot(model1.resid,model1.resid))
print('rss2', np.dot(model2.resid,model2.resid))
print('rss3', np.dot(model3.resid,model3.resid))

model1 = pd.stats.api.ols(y=test_data['price'], x=test_data[['sqft_living','bedrooms', 'bathrooms', 'lat', 'long']])
model2 = pd.stats.api.ols(y=test_data['price'], x=test_data[['sqft_living','bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']])
model3 = pd.stats.api.ols(y=test_data['price'], x=test_data[['sqft_living','bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']])

print('model 1', model1)
print('model 2', model2)
print('model 3', model3)

print('rss1', np.dot(model1.resid,model1.resid))
print('rss2', np.dot(model2.resid,model2.resid))
print('rss3', np.dot(model3.resid,model3.resid))
