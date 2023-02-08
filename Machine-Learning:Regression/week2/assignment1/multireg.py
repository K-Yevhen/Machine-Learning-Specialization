import turicreate as tc
import numpy as np

# Load in house sales data
sales = tc.SFrame("home_data.sframe")
print(sales.head())

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


