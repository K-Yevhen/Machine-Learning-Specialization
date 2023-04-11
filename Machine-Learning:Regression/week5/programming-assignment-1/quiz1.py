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

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

from sklearn.metrics import mean_squared_error

RSS_VALIDATION = []
for l1_penalty in np.logspace(1, 7, num=13):
    model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    model.fit(training[all_features], training['price'])
    prediction = model.predict(validation[all_features])
    RSS_VALIDATION.append(mean_squared_error(validation['price'], prediction))

# print("the L1 penalty which produced the lower RSS on VALIDATION: ",
#       np.logspace(1, 7, num=13)[RSS_VALIDATION.index(min(RSS_VALIDATION))])


l1_penalty_q3 = 10.0
model_q3 = linear_model.Lasso(alpha = l1_penalty_q3, normalize = True)
model_q3.fit(training[all_features], training['price'])
print("number of nonzero weights: ", (model_q3.coef_ != 0).sum() + (model_q3.intercept_ != 0).sum())


max_nonzeros = 7

def find_nonzero_weights(model):
    return (model.coef_ != 0).sum() + (model.intercept_ != 0).sum()

for l1_penalty in np.logspace(1, 4, num = 20):
    model_q4 = linear_model.Lasso(alpha = l1_penalty, normalize = True)
    model_q4.fit(training[all_features], training['price'])
    print(l1_penalty, ":", find_nonzero_weights(model_q4))


l1_penalty_min = 127.42749857
l1_penalty_max = 263.665089873


# for l1_penalty in np.linspace(l1_penalty_min, l1_penalty_max, 20):
#     model_q5 = linear_model.Lasso(alpha = l1_penalty, normalize = True)
#     model_q5.fit(training[all_features], training['price'])
#     prediction_q5 = model_q5.predict(validation[all_features])
#     RSS_q5 = mean_squared_error(prediction_q5, validation['price'])
#     print(l1_penalty, RSS_q5, find_nonzero_weights(model_q5))


model_q6 = linear_model.Lasso(alpha = 156.109096739, normalize = True)
model_q6.fit(training[all_features], training['price'])

weights_q6 = model_q6.coef_
nonzero_index_q6 = [np.where(weights_q6 == weight)[0][0] for weight in weights_q6 if weight != 0.0]
print([all_features[index] for index in nonzero_index_q6])
