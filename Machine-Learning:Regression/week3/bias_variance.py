import turicreate as tc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
MAX = sys.maxsize

tmp = tc.SArray([1., 2., 3.])
tmp_cubed = tmp.apply(lambda x: x**3)
print (tmp)
print (tmp_cubed)

ex_sframe = tc.SFrame()
ex_sframe['power_1'] = tmp
print (ex_sframe)

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

poly15_data_set1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_data_set1.column_names()
poly15_data_set1['price'] = set_1['price']
model15_1 = tc.linear_regression.create(poly15_data_set1, target = 'price', features = my_features, validation_set = None)
print(model15_1.coefficients.print_rows(16))
plt.plot(poly15_data_set1['power_1'],poly15_data_set1['price'],'.',
         poly15_data_set1['power_1'], model15_1.predict(poly15_data_set1),'-')


poly15_data_set2 = polynomial_sframe(set_2['sqft_living'], 15)
my_features = poly15_data_set2.column_names()
poly15_data_set2['price'] = set_2['price']
model15_2 = tc.linear_regression.create(poly15_data_set2, target = 'price', features = my_features, validation_set = None)
print(model15_2.coefficients.print_rows(16))
plt.plot(poly15_data_set2['power_1'],poly15_data_set2['price'],'.',
         poly15_data_set2['power_1'], model15_2.predict(poly15_data_set2),'-')


poly15_data_set3 = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly15_data_set3.column_names()
poly15_data_set3['price'] = set_3['price']
model15_3 = tc.linear_regression.create(poly15_data_set3, target = 'price', features = my_features, validation_set = None)
print(model15_2.coefficients.print_rows(16))
plt.plot(poly15_data_set3['power_1'],poly15_data_set3['price'],'.',
         poly15_data_set3['power_1'], model15_3.predict(poly15_data_set3),'-')


poly15_data_set4 = polynomial_sframe(set_4['sqft_living'], 15)
my_features = poly15_data_set4.column_names()
poly15_data_set4['price'] = set_4['price']
model15_4 = tc.linear_regression.create(poly15_data_set4, target='price', features=my_features,
                                                validation_set=None)


print(model15_4.coefficients.print_rows(16))
plt.plot(poly15_data_set4['power_1'], poly15_data_set4['price'], '.',
         poly15_data_set4['power_1'], model15_4.predict(poly15_data_set4), '-')


def getRSS(predicated, actual):
    residuals = predicated - actual
    return (residuals ** 2).sum()

def tune_model(training_set, features, validation_set):
    model = tc.linear_regression.create(training_set, target='price', features=features, validation_set=None, verbose=False)
    return getRSS(model.predict(validation_set), validation_set['price'])


training_and_validation_set, testing_set = sales.random_split(0.9, seed=1)
training_set, validation_set = training_and_validation_set.random_split(0.5, seed=1)

best_degree = 0
least_RSS = MAX
for i in range(1, 16):
    poly_i_train_set = polynomial_sframe(training_set['sqft_living'], i)
    my_features = poly_i_train_set.column_names()
    poly_i_train_set['price'] = training_set['price']

    poly_i_validation_set = polynomial_sframe(validation_set['sqft_living'], i)
    poly_i_validation_set['price'] = validation_set['price']

    RSS_of_i = tune_model(poly_i_train_set, my_features, poly_i_validation_set)
    print(i, RSS_of_i)
    if least_RSS > RSS_of_i:
        least_RSS = RSS_of_i
        best_degree = i
print('best choice is {best_degree} degree with RSS of {least_RSS}')

poly_i_train_set = polynomial_sframe(training_set['sqft_living'], 6)
my_features = poly_i_train_set.column_names()
poly_i_train_set['price'] = training_set['price']

poly_i_test_set = polynomial_sframe(testing_set['sqft_living'], 6)
poly_i_test_set['price'] = testing_set['price']

model = tc.linear_regression.create(poly_i_train_set, target='price', features=my_features, validation_set=None,
                                            verbose=False)

print(getRSS(model.predict(poly_i_test_set),testing_set['price']))
