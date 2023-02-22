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
# model = polynomial_ridge_regression(data, deg=16, l2_penalty=100)
# print_coefficients(model)
# plot_poly_predictions(data, model)

# Let's look at fits for a sequence of increasing lambda values
# for l2_penalty in [1e-25, 1e-10, 1e-6, 1e-3, 1e2]:
#     model = polynomial_ridge_regression(data, deg=12, l2_penalty=l2_penalty)
#     print 'lambda = %.2e' % l2_penalty
#     print_coefficients(model)
#     print '\n'
#     plt.figure()
#     plot_poly_predictions(data, model)
#     plt.title('Ridge, lambda = %.2e' % l2_penalty)

# Perform a ridge fit of a degree-16 polynomial using a "good" penalty strength
# LOO cross validation -- return the average MSE
from sklearn import model_selection as ms

def loo(data, deg, l2_penalty_values):
    # Create polynomial features
    polynomial_features(data, deg)

    # Create as many folds for cross validation as number of data point
    num_folds = len(data)
    folds = ms.KFold(data, num_folds)

    # for each value of l2_penalty, fit a model for each fold and compute average MSe
    l2_penalty_mse = []
    min_mse = None
    best_l2_penalty = None
    for l2_penalty in l2_penalty_values:
        next_mse = 0.0
        for train_set, validation_set in folds:
            # train model
            model = tc.linear_regression.create(train_set, target='Y',
                                                l2_penalty=l2_penalty,
                                                validation_set=None, verbose=False)

            # predict on validation set
            y_test_predicted = model.predict(validation_set)
            # compute squared error
            next_mse += ((y_test_predicted - validation_set['Y'])**2).sum()

        # save squared error in list of MSE for each l2_penalty
        next_mse = next_mse / num_folds
        l2_penalty_mse.append(next_mse)
        if min_mse in None or next_mse < min_mse:
            min_mse = next_mse
            best_l2_penalty = l2_penalty

    return l2_penalty_mse, best_l2_penalty

# Run LOO cross validation for "num" values of lambda, on a log scale

l2_penalty_valeus = numpy.logspace(-4, 10, num=10)
l2_penalty_mse, best_l2_penalty = loo(data, 16, l2_penalty_valeus)

# Plot results of estimating LOO for each value of lambda
plt.plot(l2_penalty_valeus, l2_penalty_mse, 'k-')
plt.xlabel('$\l2_penalty$')
plt.ylabel('LOO cross validation error')
plt.xscale('log')
plt.yscale('log')

print(best_l2_penalty)
model = polynomial_ridge_regression(data, deg=16, l2_penalty=best_l2_penalty)
print_coefficients(model)
plot_poly_predictions(data, model)
