import numpy as np
import turicreate as tc
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


sales = tc.SFrame('home_data.sframe')
# print(sales.head())


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    feature_matrix = data_sframe[features].to_numpy()
    output_array = data_sframe[output].to_numpy()
    return (feature_matrix, output_array)


def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return (predictions)

