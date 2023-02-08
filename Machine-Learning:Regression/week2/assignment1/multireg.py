import turicreate as tc
import numpy as np
from math import log

# Load in house sales data
sales = tc.SFrame("home_data.sframe")
# print(sales.head())

# split data into training and testing
train_data, test_data = sales.random_split(.8, seed=0)

# Learning a multiple regression model
"""Recall we can use the following code to learn a multiple regression model predicting 'price' based on the following features:
example_features = ['sqft_living', 'bedrooms', 'bathrooms'] on training data with the following code:
(Aside: We set validation_set = None to ensure that the results are always the same)"""

# example_features = ['sqft_living', 'bedrooms', 'bathrooms']
# example_model = tc.linear_regression.create(train_data, target='price', features=example_features,
#                                             validation_set=None)

bedrooms_squared = test_data['bedrooms'] * test_data['bedrooms']
bed_bath_rooms = test_data['bedrooms'] * test_data['bathrooms']
log_sqft_living = test_data['sqft_living'].apply(np.log)
lat_plus_long = test_data['lat'] + test_data['long']

# Question 1
print(bedrooms_squared.mean())

# Question 2
print(bed_bath_rooms.mean())

# Question 3
print(log_sqft_living.mean())

# Question 4
print(lat_plus_long.mean())

train_data['bedrooms_squared'] = train_data['bedrooms'] ** 2
train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
train_data['log_sqft_living'] = train_data['sqft_living'].apply(np.log)
train_data['lat_plus_long'] = train_data['lat'] + train_data['long']

test_data['bedrooms_squared'] = test_data['bedrooms'] ** 2
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']
test_data['log_sqft_living'] = test_data['sqft_living'].apply(np.log)
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

model_1 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']
model_3 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

bathroom_model_1 = tc.linear_regression.create(train_data, target='price', features=model_1,
                                             validation_set=None)
# Question 5
print(bathroom_model_1.coefficients)

bathroom_model_2 = tc.linear_regression.create(train_data, target='price', features=model_2,
                                             validation_set=None)
# Question 6
print(bathroom_model_2.coefficients)

bathroom_model_3 = tc.linear_regression.create(train_data, target='price', features=model_3,
                                             validation_set=None)
# Question 7
print(bathroom_model_1.predict) # Training RMSE                  : 236378.5965
print(bathroom_model_2.predict) # Training RMSE                  : 235190.9354
print(bathroom_model_3.predict) # Training RMSE                  : 228200.0432

bathroom_model_1 = tc.linear_regression.create(test_data, target='price', features=model_1,
                                             validation_set=None)
bathroom_model_2 = tc.linear_regression.create(test_data, target='price', features=model_2,
                                             validation_set=None)
bathroom_model_3 = tc.linear_regression.create(test_data, target='price', features=model_3,
                                             validation_set=None)

print(bathroom_model_1.predict) # Training RMSE                  : 232501.847
print(bathroom_model_2.predict) # Training RMSE                  : 230924.0261
print(bathroom_model_3.predict) # Training RMSE                  : 225678.0117
