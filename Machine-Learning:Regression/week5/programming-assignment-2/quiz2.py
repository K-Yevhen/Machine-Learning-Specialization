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

