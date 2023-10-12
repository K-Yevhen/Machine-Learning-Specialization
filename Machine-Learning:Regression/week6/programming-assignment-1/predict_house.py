import numpy as np
import turicreate as tc
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

# for better working with Pandas library
dtype_dict = {'bathrooms': float,
              'waterfront': int,
              'sqft_above': int,
              'sqft_living15': float,
              'grade': int,
              'yr_renovated': int,
              'price': float,
              'bedrooms': float,
              'zipcode': str,
              'long': float,
              'sqft_lot15': float,
              'sqft_living': float,
              'floors': float,
              'condition': int,
              'lat': float,
              'date': str,
              'sqft_basement': int,
              'yr_built': int,
              'id': str,
              'sqft_lot': int,
              'view': int}

sales = tc.SFrame("/Users/yevhenkuts/PycharmProjects/New/Machine-Learning-Specialization/Machine-Learning:Regression/week6/programming-assignment-1/home_data_small.sframe")

(train_and_validation, test) = sales.random_split(.8, seed=1)
(train, validation) = train_and_validation.random_split(.8, seed=1)

sales_pandas = pd.read_csv('kc_house_data_small.csv', dtype=dtype_dict)
training_pandas = pd.read_csv('kc_house_data_small_train.csv', dtype=dtype_dict)
testing_pandas = pd.read_csv('kc_house_data_small_test.csv', dtype=dtype_dict)
validation_pandas = pd.read_csv('kc_house_data_validation.csv', dtype=dtype_dict)

print(sales_pandas.head())