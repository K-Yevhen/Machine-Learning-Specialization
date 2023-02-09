# coding=utf-8
import turicreate as tc
import numpy as np

# Exploring the data
sales = tc.SFrame("home_data.sframe")
# print(sales)


#3  Next write a function that takes a data set, a list of features (e.g. [‘sqft_living’, ‘bedrooms’]), to be used as inputs
def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1  # add a constant column to an DataFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features

    features_dataframe = data_frame[features]

    features_matrix = features_dataframe.as_matrix()

    output_dataframe = data_frame[output]
    output_array = output_dataframe.as_matrix()

    return features_matrix, output_array


# 4
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions


# 5
def feature_derivative(errors, feature):
    derivative = 2 * np.dot(feature, errors)
    return derivative


# train_data,test_data = sales.random_split(.8,seed=0)
