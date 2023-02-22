import numpy as np
import turicreate as tc
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


sales = tc.SFrame('home_data.sframe')
# print(sales.head())

train_data, test_data = sales.random_split(.8, seed=0)

def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    feature_matrix = data_sframe[features].to_numpy()
    output_array = data_sframe[output].to_numpy()
    return feature_matrix, output_array


def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant = False):
    if not feature_is_constant:
        derivative = 2 * np.dot(feature, errors) + 2 * l2_penalty * weight
    else:
        derivative = 2 * np.dot(feature, errors)
    return derivative


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations=100):
    converged = False
    weights = np.array(initial_weights)  # make sure it's a numpy array
    # while not reached maximum number of iterations:

    while max_iterations > 0:
        # compute the predictions using your predict_output() function
        predictions = predict_output(feature_matrix, weights)

        # compute the errors as predictions - output
        errors = predictions - output

        for i in range(len(weights)):  # loop over each weight

            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            if i == 0:
                feature_is_constant = True
            else:
                feature_is_constant = False
            derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty,
                                                  feature_is_constant)
            # (Remember: when i=0, you are computing the derivative of the constant!)

            # subtract the step size times the derivative from the current weight
            weights[i] -= step_size * derivative
        max_iterations -= 1
    return weights

simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, [0.0,0.0], 1e-12, 0.0, max_iterations=1000)
print(simple_weights_0_penalty)

simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, [0.0,0.0], 1e-12, 1e11, max_iterations=1000)
print(simple_weights_high_penalty)


plt.plot(simple_feature_matrix,output,'k.',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')
plt.show()


