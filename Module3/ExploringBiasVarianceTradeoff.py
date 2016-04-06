import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def polynomial_dataframe(feature, degree):
    poly_dataframe = pd.DataFrame()
    poly_dataframe['power_1'] = feature

    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_dataframe[name] = np.power(feature,power)

    return(poly_dataframe)

def estimate_deg_model(data_set, deg):
    data_set = data_set.sort(['sqft_living','price'])

    poly_data = polynomial_dataframe(data_set['sqft_living'], deg)
    poly_data['price'] = data_set['price']

    model = pd.stats.api.ols(y=poly_data['price'], x=poly_data[poly_data.columns -['price']])

    #print('Weights')
    #print(model.beta)

    plt.plot(poly_data['power_1'], poly_data['price'], '.',
    poly_data['power_1'], model.y_predict, '-')
    #plt.show()
    return(model)

def estimate_15_deg_model(data_set):
    return(estimate_deg_model(data_set, 15))


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('Data_wk3/kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])

poly1_data = polynomial_dataframe(sales['sqft_living'], 15)
poly1_data['price'] = sales['price']

model1 = pd.stats.api.ols(y=poly1_data['price'], x=poly1_data[poly1_data.columns -['price']])

print('Weights')
print(model1.beta)

plt.plot(poly1_data['power_1'], poly1_data['price'], '.',
poly1_data['power_1'], model1.y_predict, '-')
plt.show()

set_1 = pd.read_csv('Data_wk3/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('Data_wk3/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('Data_wk3/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('Data_wk3/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

print('set1')
model1 = estimate_15_deg_model(set_1)
print('set2')
model2 = estimate_15_deg_model(set_2)
print('set3')
model3 = estimate_15_deg_model(set_3)
print('set4')
model4 = estimate_15_deg_model(set_4)

train_set = pd.read_csv('Data_wk3/wk3_kc_house_train_data.csv', dtype=dtype_dict)
valid_set = pd.read_csv('Data_wk3/wk3_kc_house_valid_data.csv', dtype=dtype_dict)
test_set = pd.read_csv('Data_wk3/wk3_kc_house_test_data.csv', dtype=dtype_dict)

error = list()
for deg in range(1, 16):
    model = estimate_deg_model(train_set, deg)
    predict_set = polynomial_dataframe(valid_set['sqft_living'],deg)
    predict_values = model.predict(x=predict_set)
    error.append(np.sum(np.square(valid_set['price'] - predict_values)))


#correct answer should be min_deg = 6...?
print('argmin 1-based')
print(np.argmin(error)+1)

min_deg = np.argmin(error)+1

best_model = estimate_deg_model(train_set, min_deg)
check_set = polynomial_dataframe(test_set['sqft_living'],min_deg)
check_values = best_model.predict(x=check_set)

print('test rss')
print(np.sum(np.square(test_set['price'] - check_values)))
    



