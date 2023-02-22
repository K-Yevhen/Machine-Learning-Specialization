# Create a dataset based on true sinusoidal relationship
import numpy
import turicreate as tc
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Create random values for x in internal [0,1)

random.seed(98103)
n = 30
x = tc.SArray([random.random() for i in range(n)]).sort()

# Compute y
y = x.apply(lambda x: math.sin(4*x))

# Add random Gaussian noise to y
random.seed(1)
e = tc.SArray([random.gauss(0,1.0/3.0) for i in range(n)])
y = y + e

# Pit data into an SFrame to manipulate later
data = tc.SFrame({'X1': x, 'Y': y})
# print(data)

# Create a function to plot the data, since we will do in many times


def plot_data(data):
    plt.plot(data['X1'], data['Y'], 'k.')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()
# plot_data(data)

# Define some useful polynomial regression functions


def polynomial_features(data, deg):
    data_copy = data.copy()
    for i in range(1, deg):
        data_copy['X'+str(i+1)] = data_copy['X'+str(i)]*data_copy['X1']
    return data_copy

# Define a function to fit a polynomial linear regression model of degree "deg" to the data in "data"


def polynomial_regression(data, deg):
    model = tc.linear_regression.create(
        polynomial_features(data, deg),
        target='Y',
        l2_penalty=0.,
        l1_penalty=0.,
        validation_set=None,
        verbose=False)
    return model

# Define function to plat data and predictions made, since we are going to use it many times


def plot_poly_predictions(data, model):
    plot_data(data)

    # Get the degree of the polynomial
    deg = len(model.coefficients['value'])-1

    # Create 200 points in the x axis and compute the predicted value for each point
    x_pred = tc.SFrame({'X1': [i / 200.0 for i in range(200)]})
    y_pred = model.predict(polynomial_features(x_pred, deg))

    # plot predictions
    plt.plot(x_pred['X1'], y_pred, 'g-', label='degree ' + str(deg) + ' fit')
    plt.legend(loc='upper left')
    plt.axis([0, 1, -1.5, 2])
    plt.show()

# Create a function that prints the polynomial coefficients in a pretty way


def print_coefficients(model):
    # Get the degree of the polynomial
    deg = len(model.coefficients['value'])-1

    # Get learned parameters as a list
    w = list(model.coefficients['value'])

    # Numpy has a nifty function to print out polynomials in a pretty way
    # (We'll use it, but it needs the parameters in the reverse order)
    print('Learned polynomial for degree ' + str(deg) + ':')
    w.reverse()
    print(numpy.poly1d(w))

# Fit a degree-2 polynomial
# model = polynomial_regression(data, deg=2)
# print_coefficients(model)
# plot_poly_predictions(data, model)

# Fit a degree-4 polynomial
# model = polynomial_regression(data, deg=4)
# print_coefficients(model)
# plot_poly_predictions(data, model)

# Fit a degree-16 polynomial
# model = polynomial_regression(data, deg=16)
# print_coefficients(model)
# plot_poly_predictions(data, model)

# Ridge Regression


def polynomial_ridge_regression(data, deg, l2_penalty):
    model = tc.linear_regression.create(polynomial_features(data, deg),
                                        target='Y', l2_penalty=l2_penalty,
                                        validation_set=None, verbose=False)
    return model

# Perform a ridge fit of a degree-16 polynomial using a very small penalty strength

# model = polynomial_ridge_regression(data, deg=16, l2_penalty=1e-25)
# print_coefficients(model)
# plot_poly_predictions(data, model)

# Perform a ridge fit of a degree-16 polynomial using a very large penalty strength
model = polynomial_ridge_regression(data, deg=16, l2_penalty=100)
print_coefficients(model)
plot_poly_predictions(data, model)
