import pandas as pd
import numpy as np
import turicreate as tc
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from sklearn import linear_model
from scipy import sqrt
from sklearn.metrics import mean_squared_error


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

house_train = pd.read_csv('wk3_kc_house_train_data.csv', dtype = dtype_dict)
house_train = house_train.sort_values(['sqft_living','price'])
house_test = pd.read_csv('wk3_kc_house_test_data.csv', dtype = dtype_dict)
house_test = house_test.sort_values(['sqft_living','price'])
house_valid = pd.read_csv('wk3_kc_house_valid_data.csv', dtype = dtype_dict)
house_valid = house_valid.sort_values(['sqft_living','price'])

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort_values(['sqft_living','price'])

# print(house_train.head())


l2_small_penalty = 1.5e-5

def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = poly_dataframe['power_1'].apply(lambda x: x**power)
    return poly_dataframe



poly15_train_data = polynomial_dataframe(sales['sqft_living'], 15)

model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(poly15_train_data, sales['price'])

# Question 1
print("Question 1: {}".format(model.coef_[0]))

# Question 2
set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

def fit_polynomial_model(set_data, l2_small_penalty):
    poly15_set_data = polynomial_dataframe(set_data['sqft_living'], 15)
    model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model.fit(poly15_set_data, set_data['price'])
    return model

l2_small_penalty=1e-9

set_1_model = fit_polynomial_model(set_1, l2_small_penalty)
print('the coefficient of feature power_1 for set_1 is: ', set_1_model.coef_[0])
set_2_model = fit_polynomial_model(set_2, l2_small_penalty)
print('the coefficient of feature power_1 for set_2 is: ', set_2_model.coef_[0])
set_3_model = fit_polynomial_model(set_3, l2_small_penalty)
print('the coefficient of feature power_1 for set_3 is: ', set_3_model.coef_[0])
set_4_model = fit_polynomial_model(set_4, l2_small_penalty)
print('the coefficient of feature power_1 for set_4 is: ', set_4_model.coef_[0])

# Question 3
print('the coefficient of feature power_1 for set_4 is: ', set_4_model.coef_[0])
l2_large_penalty=1.23e2
set_1_model_2 = fit_polynomial_model(set_1, l2_large_penalty)
print('the coefficient of feature power_1 for set_1 is: ', set_1_model_2.coef_[0])
set_2_model_2 = fit_polynomial_model(set_2, l2_large_penalty)
print('the coefficient of feature power_1 for set_2 is: ', set_2_model_2.coef_[0])
set_3_model_2 = fit_polynomial_model(set_3, l2_large_penalty)
print('the coefficient of feature power_1 for set_3 is: ', set_3_model_2.coef_[0])
set_4_model_2 = fit_polynomial_model(set_4, l2_large_penalty)
print('the coefficient of feature power_1 for set_4 is: ', set_4_model_2.coef_[0])


# Question 4
print('the coefficient of feature power_1 for set_4 is: ', set_4_model_2.coef_[0])

# Question 5

# Question 6

# Question 7
poly15_house_train = polynomial_dataframe(house_train['sqft_living'], 15)
poly15_house_test = polynomial_dataframe(house_test['sqft_living'], 15)

model_q7 = linear_model.Ridge(alpha = 1000, normalize = True)
model_q7.fit(poly15_house_train, house_train['price'])

prediction_q7 = model_q7.predict(poly15_house_test)
RSS_q7 = sqrt(mean_squared_error(house_test['price'], prediction_q7))

print("the RSS on the TEST data is: ", RSS_q7)
