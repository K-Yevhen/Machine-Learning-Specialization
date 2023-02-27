import turicreate as tc
import numpy
import turicreate as tc
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def polynomial_features(data, deg):
    data_copy = data.copy()
    for i in range(1, deg):
        data_copy['X'+str(i+1)] = data_copy['X'+str(i)]*data_copy['X1']
    return data_copy


def polynomial_lasso_regression(data, deg, l1_penalty):
    model = tc.linear_regression.create(polynomial_features(data,deg),
                                                target='Y', l2_penalty=0.,
                                                l1_penalty=l1_penalty,
                                                validation_set=None,
                                                solver='fista', verbose=False,
                                                max_iterations=3000, convergence_threshold=1e-10)
    return model