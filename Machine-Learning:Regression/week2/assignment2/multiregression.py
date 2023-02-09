# coding=utf-8
import turicreate as tc
import numpy as np
from math import sqrt

# Exploring the data
sales = tc.SFrame("home_data.sframe")
# print(sales)


# 3 Next write a function that takes a data set, a list of features (e.g. [‘sqft_living’, ‘bedrooms’])
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


# 4
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions


# 5
def feature_derivative(errors, feature):
    derivative = 2 * np.dot(feature, errors)
    return derivative


# 6. Now we will use our predict_output and feature_derivative to write a gradient descent function.
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        errors = predictions - output

        gradient_sum_squares = 0  # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:, i])

            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += derivative ** 2

            # update the weight based on step size and derivative:
            weights[i] -= step_size * derivative

        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return weights


train_data, test_data = sales.random_split(.8, seed=0)

simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

# Question - 1
simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
print(simple_weights[1])

# Question - 2
house_1st = test_data['sqft_living'][0]
prediction_house_1st = predict_outcome(np.array([1, house_1st]), simple_weights)
print(prediction_house_1st)

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

# Question - 3
my_predicted_house_2 = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print(my_predicted_house_2)
(feature_matrix, output) = get_numpy_data(test_data, model_features, 'price')
my_predicted_values_on_test_2 = predict_outcome(np.array(feature_matrix), my_predicted_house_2)
print(my_predicted_values_on_test_2[0])

# Question - 4
print('True price: ', test_data['price'][0])
print('Model 1 price: ', prediction_house_1st)
print('Model 2 price: ', my_predicted_values_on_test_2[0])

# Question - 5  - failed
