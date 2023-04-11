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

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales_panda = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)


sales = tc.SFrame('home_data.sframe')
# print(sales.head())

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors'] = sales['floors'].astype(float)
sales['floors_square'] = sales['floors']*sales['floors']

# print(sales['sqft_living_sqrt'])

from sklearn import linear_model  # using scikit-learn
all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = tc.linear_regression.create(sales, target='price', features=all_features, validation_set=None, l2_penalty=0.,
                                        l1_penalty=1e10)

# print(model_all.coefficients)

zero = model_all.coefficients
nonzero_coefficients = zero[zero['value'] != 0]
# print(nonzero_coefficients)

(training_and_validation, testing) = sales.random_split(.9, seed=1)
(training, validation) = training_and_validation.random_split(0.5, seed=1)


