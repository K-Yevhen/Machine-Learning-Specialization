import turicreate as tc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = tc.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_sframe[name] to be feature^power
            poly_sframe[name] = poly_sframe['power_1'].apply(lambda x: x**power)
    return poly_sframe

sales = tc.SFrame('home_data.sframe')
sales = sales.sort(['sqft_living', 'price'])

poly1_data = polynomial_sframe(sales['sqft_living'], 1)
# print(poly1_data.head())

poly1_data['price'] = sales['price']
# print(poly1_data.head())

model1 = tc.linear_regression.create(poly1_data, target='price', features=['power_1'], validation_set=None)

# plt.plot(poly1_data['power_1'], poly1_data['price'], '.', poly1_data['power_1'], model1.predict(poly1_data), '-')
# plt.show()

poly2_data = polynomial_sframe(sales['sqft_living'], 2)
poly2_data['price'] = sales['price']

model2 = tc.linear_regression.create(poly2_data, target='price', features=['power_1', 'power_2'], validation_set=None)
# plt.plot(poly2_data['power_1'], poly2_data['price'], '.', poly2_data['power_1'], model2.predict(poly2_data[['power_1', 'power_2']]),'-')
# plt.show()

poly3_data = polynomial_sframe(sales['sqft_living'], 3)
poly3_data['price'] = sales['price']

model3 = tc.linear_regression.create(poly3_data, target='price', features=['power_1', 'power_2', 'power_3'], validation_set=None)
# plt.plot(poly3_data['power_1'], poly3_data['price'], '.', poly3_data['power_1'], model3.predict(poly3_data[['power_1', 'power_2', 'power_3']]), '-')
# plt.show()

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = sales['price']
model15 = tc.linear_regression.create(poly15_data, target='price', features=my_features, validation_set=None)
# plt.plot(poly15_data['power_1'], poly15_data['price'], '.',
#          poly15_data['power_1'], model15.predict(poly15_data), '-')
# plt.show()

set_1, set_3 = sales.random_split(0.5, seed=0)
set_1, set_2 = set_1.random_split(0.5, seed=0)
set_3, set_4 = set_3.random_split(0.5, seed=0)

