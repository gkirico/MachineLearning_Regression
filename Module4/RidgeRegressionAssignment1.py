import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def polynomial_dataframe(feature, degree):
    poly_dataframe = pd.DataFrame()
    poly_dataframe['power_1'] = feature

    if degree > 1:
        for power in range(2, degree + 1):
            name = 'power_' + str(power)
            poly_dataframe[name] = np.power(feature,power)

    return(poly_dataframe)

def k_fold_cross_validation(k, l2_penalty, data, output):
    n = len(data)
    error = 0
    for i in xrange(k):
        start = (n * i) / k
        end = (n * (i + 1)) / k - 1
        val_data = data[start:end + 1]
        val_output = output[start:end + 1]
        train_data = data[0:start].append(data[end + 1:n])
        train_output = output[0:start].append(output[end + 1:n])

        model = linear_model.Ridge(alpha=l2_penalty, normalize=True)
        model.fit(train_data, train_output)

        error += np.sum(np.square(model.predict(val_data) - val_output))

    avg_validation_error = error/k
    return(avg_validation_error)

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('Data_wk4/kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])

l2_small_penalty = 1.5e-5

poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(poly15_data, sales['price'])

print('coeff power_1', model.coef_[0])

# dtype_dict same as above
set_1 = pd.read_csv('Data_wk4/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('Data_wk4/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('Data_wk4/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('Data_wk4/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

poly15_set1 = polynomial_dataframe(set_1['sqft_living'], 15)
poly15_set2 = polynomial_dataframe(set_2['sqft_living'], 15)
poly15_set3 = polynomial_dataframe(set_3['sqft_living'], 15)
poly15_set4 = polynomial_dataframe(set_4['sqft_living'], 15)

l2_small_penalty = 1e-9

model1 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model1.fit(poly15_set1, set_1['price'])
print('coeff power_1 set 1', model1.coef_[0])

model2 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model2.fit(poly15_set2, set_2['price'])
print('coeff power_1 set 2', model2.coef_[0])

model3 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model3.fit(poly15_set3, set_3['price'])
print('coeff power_1 set 3', model3.coef_[0])

model4 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model4.fit(poly15_set4, set_4['price'])
print('coeff power_1 set 4', model4.coef_[0])

#plt.hold()
#plt.plot(poly15_set1['power_1'], set_1['price'], 'r.',
#    poly15_set1['power_1'], model1.predict(poly15_set1), 'r-')

#plt.plot(poly15_set2['power_1'], set_2['price'], 'b.',
#    poly15_set2['power_1'], model2.predict(poly15_set2), 'b-')

#plt.plot(poly15_set3['power_1'], set_3['price'], 'g.',
#    poly15_set3['power_1'], model3.predict(poly15_set3), 'g-')

#plt.plot(poly15_set4['power_1'], set_4['price'], 'c.',
#    poly15_set4['power_1'], model4.predict(poly15_set4), 'c-')
#plt.show()
print('ridge regression')

l2_large_penalty = 1.23e2

model1 = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)
model1.fit(poly15_set1, set_1['price'])
print('coeff power_1 set 1', model1.coef_[0])

model2 = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)
model2.fit(poly15_set2, set_2['price'])
print('coeff power_1 set 2', model2.coef_[0])

model3 = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)
model3.fit(poly15_set3, set_3['price'])
print('coeff power_1 set 3', model3.coef_[0])

model4 = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)
model4.fit(poly15_set4, set_4['price'])
print('coeff power_1 set 4', model4.coef_[0])

train_valid_shuffled = pd.read_csv('Data_wk4/wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('Data_wk4/wk3_kc_house_test_data.csv', dtype=dtype_dict)

k = 10 # 10-fold cross-validation
poly15_shuffled = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)

for l2_penalty in np.logspace(3,9, num=13):
    print(l2_penalty, k_fold_cross_validation(k, l2_penalty, poly15_shuffled, train_valid_shuffled['price']))


poly15_test = polynomial_dataframe(test['sqft_living'], 15)

bestmodel = linear_model.Ridge(alpha=1000, normalize=True)
bestmodel.fit(poly15_shuffled, train_valid_shuffled['price'])

print('RSS test', np.sum(np.square(bestmodel.predict(poly15_test) - test['price'])))
