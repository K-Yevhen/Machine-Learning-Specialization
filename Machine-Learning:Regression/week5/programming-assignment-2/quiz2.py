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
import math

sales_tc = tc.SFrame('home_data.sframe')

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales_pd = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
training_pd = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
testing_pd = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)


sales_tc['floors'] = sales_tc['floors'].astype(float).astype(int)


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1  # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features  # this is how you combine two lists

    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = data_sframe[features].to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = data_sframe[output].to_numpy()
    return feature_matrix, output_array



def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    return feature_matrix / norms, norms

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales_pd, simple_features, my_output)

(normalized_simple_feature_matrix, norms) = normalize_features(simple_feature_matrix)
initial_weights = np.array([1,4,1])
prediction = predict_output(normalized_simple_feature_matrix, initial_weights)

ro = [0 for i in range(normalized_simple_feature_matrix.shape[1])]

for j in range(normalized_simple_feature_matrix.shape[1]):
    ro[j] = (normalized_simple_feature_matrix[:, j] * (
                output - prediction + initial_weights[j] * normalized_simple_feature_matrix[:, j])).sum()

print(ro)
print(np.array(ro) * 2)


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = (feature_matrix[:, i] * (output - prediction + weights[i] * feature_matrix[:, i])).sum()

    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2.0
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2.0
    else:
        new_weight_i = 0.

    return new_weight_i

print(lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]), np.array([1., 1.]), np.array([1., 4.]), 0.1))


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    D = feature_matrix.shape[1]
    weights = np.array(initial_weights)
    weight_change = np.array(initial_weights) * 0.0
    converged = False

    while not converged:
        for index in range(D):
            new_weight = lasso_coordinate_descent_step(index, feature_matrix, output, weights, l1_penalty)

            weight_change[index] = np.abs(new_weight - weights[index])

            weights[index] = new_weight

        if max(weight_change) < tolerance:
            converged = True

    return (weights)

initial_weights = np.zeros(simple_feature_matrix.shape[1])
l1_penalty = 1e7
tolerance = 1.0

weights_q3 = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output, initial_weights, l1_penalty, tolerance)

def RSS_calculation(feature_matrix, weights, output):
    prediction = predict_output(feature_matrix, weights)
    residual = output - prediction
    RSS = (residual ** 2).sum()
    return RSS

print(RSS_calculation(normalized_simple_feature_matrix, weights_q3, output))
print(weights_q3)


more_features = ['bedrooms',
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
                'yr_renovated']

(more_feature_matrix, more_feature_output) = get_numpy_data(training_pd, more_features, my_output)

(normalized_more_feature_matrix, more_feature_norms) = normalize_features(more_feature_matrix)

initial_weights = np.zeros(normalized_more_feature_matrix.shape[1])
l1_penalty = 1e7
tolerance = 1.0

weights1e7 = lasso_cyclical_coordinate_descent(normalized_more_feature_matrix, more_feature_output, initial_weights, l1_penalty, tolerance)
