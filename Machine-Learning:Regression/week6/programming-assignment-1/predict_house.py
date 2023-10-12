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

sales = tc.SFrame("/Users/yevhenkuts/PycharmProjects/New/Machine-Learning-Specialization/"
                  "Machine-Learning:Regression/week6/programming-assignment-1/home_data_small.sframe")

(train_and_validation, test) = sales.random_split(.8, seed=1)
(train, validation) = train_and_validation.random_split(.8, seed=1)

sales_pandas = pd.read_csv('/Users/yevhenkuts/PycharmProjects/New/Machine-Learning-Specialization/'
                           'Machine-Learning:Regression/week6/programming-assignment-1/'
                           'kc_house_data_small.csv', dtype=dtype_dict)
training_pandas = pd.read_csv('/Users/yevhenkuts/PycharmProjects/New/Machine-Learning-Specialization/'
                              'Machine-Learning:Regression/week6/programming-assignment-1/'
                              'kc_house_data_small_train.csv', dtype=dtype_dict)
testing_pandas = pd.read_csv('/Users/yevhenkuts/PycharmProjects/New/Machine-Learning-Specialization/'
                             'Machine-Learning:Regression/week6/programming-assignment-1/'
                             'kc_house_data_small_test.csv', dtype=dtype_dict)
validation_pandas = pd.read_csv('/Users/yevhenkuts/PycharmProjects/New/Machine-Learning-Specialization/'
                                'Machine-Learning:Regression/week6/programming-assignment-1/'
                                'kc_house_data_validation.csv', dtype=dtype_dict)

# print(sales_pandas.head())


def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1  # add a constant column to an DataFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features

    features_dataframe = data_frame[features]

    features_matrix = features_dataframe.values

    output_dataframe = data_frame[output]
    output_array = output_dataframe.values

    return (features_matrix, output_array)


def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms
    return(normalized_features, norms)


all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated',
                'lat',
                'long',
                'sqft_living15',
                'sqft_lot15']

my_output = 'price'

(feature_train, output_train) = get_numpy_data(training_pandas, all_features, my_output)
(feature_train, norms) = normalize_features(feature_train)

(feature_test, output_test) = get_numpy_data(testing_pandas, all_features, my_output)
(feature_valid, output_valid) = get_numpy_data(validation_pandas, all_features, my_output)

feature_test = feature_test / norms
feature_valid = feature_valid / norms

# print("Features test for 0: {}".format(feature_test[0]))
# print("Features test for 9: {}".format(feature_test[9]))

# Question 1
# print(np.sqrt(((feature_test[0] - feature_train[9]) ** 2).sum()))


def compute_distances(features_instances, features_query):
    diff = features_instances - features_query
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return distances


distances_3house = compute_distances(feature_train, feature_test[2])

# Question 3
# print(np.argmin(distances_3house))

# Question 4
print(training_pandas['price'][382])
