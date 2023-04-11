import pandas as pd
import numpy as np
from math import log, sqrt
from sklearn import linear_model
from scipy import sqrt
from sklearn.metrics import mean_squared_error

import turicreate as tc
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

# dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
# sales_panda = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
#
#
# sales = tc.SFrame('home_data.sframe')
# print(sales.head())
#
# sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
# sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
# sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
# sales['floors'] = sales['floors'].astype(float)
# sales['floors_square'] = sales['floors']*sales['floors']
#
# print(sales['sqft_living_sqrt'])
#
# from sklearn import linear_model  # using scikit-learn
# all_features = ['bedrooms', 'bedrooms_square',
#             'bathrooms',
#             'sqft_living', 'sqft_living_sqrt',
#             'sqft_lot', 'sqft_lot_sqrt',
#             'floors', 'floors_square',
#             'waterfront', 'view', 'condition', 'grade',
#             'sqft_above',
#             'sqft_basement',
#             'yr_built', 'yr_renovated']
#
# model_all = tc.linear_regression.create(sales, target='price', features=all_features, validation_set=None, l2_penalty=0.,
#                                         l1_penalty=1e10)
#
# print(model_all.coefficients)
#
# zero = model_all.coefficients
# nonzero_coefficients = zero[zero['value'] != 0]
# print(nonzero_coefficients)
#
# (training_and_validation, testing) = sales.random_split(.9, seed=1)
# (training, validation) = training_and_validation.random_split(0.5, seed=1)
#
# l1_penalties = np.logspace(1, 7, num=13)
# best_l1 = -1
# least_rss = -1
# for l1 in l1_penalties:
#     model = tc.linear_regression.create(dataset=training, target='price', features=all_features,
#                                                 l2_penalty=0, l1_penalty=l1, validation_set=None, verbose=False)
#     predicted = model.predict(validation)
#     cur_rss = sum(pow(predicted - validation['price'], 2))
#
#     if least_rss == -1 or cur_rss < least_rss:
#         least_rss = cur_rss
#         best_l1 = l1
#     print(l1, cur_rss)
#
# model = tc.linear_regression.create(dataset=training, target='price', features=all_features,
#                                             l2_penalty=0, l1_penalty=best_l1, validation_set=None, verbose=False)
#
# print(model.coefficients['value'].nnz())

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype = dtype_dict)
testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype = dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype = dtype_dict)
# print(sales.head())

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha=5e2, normalize=True)  # set parameters
print(model_all.fit(sales[all_features], sales['price']))  # learn weights

weights_all = model_all.coef_
non_zero_index = [np.where(weights_all == weight)[0][0] for weight in weights_all if weight != 0.0]
selected_features = [all_features[index] for index in non_zero_index]
# print(selected_features)

