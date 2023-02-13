import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long':float, 'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int, 'yr_built': int, 'id':str, 'sqft_lot': int, 'view': int}

# house_train = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
# house_test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
# house_valid = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)

# print(house_train.head(20))

def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = poly_dataframe['power_1'].apply(lambda x: x ** power)
    return poly_dataframe

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort_values(['sqft_living', 'price'])

poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']


poly1_data['price'] = sales['price']
# print(poly1_data.head(20))


model1 = linear_model.LinearRegression().fit(np.array([poly1_data['power_1']]).T, np.array([poly1_data['price']]).T)
intercept_1 = model1.intercept_[0]
slope_1 = model1.coef_[0][0]
print("intercept: ", intercept_1)
print("slope: ", slope_1)

plt.plot(poly1_data['power_1'], poly1_data['price'], '.', poly1_data['power_1'], model1.predict(np.array([poly1_data['power_1']]).T), '-')
plt.show()

poly2_data = polynomial_dataframe(sales['sqft_living'], 2)
poly2_data['price'] = sales['price']

model2 = linear_model.LinearRegression().fit(poly2_data[['power_1', 'power_2']], np.array([poly2_data['price']]).T)
plt.plot(poly2_data['power_1'], poly2_data['price'], '.', poly2_data['power_1'], model2.predict(poly2_data[['power_1','power_2']]),'-')
plt.show()

poly3_data = polynomial_dataframe(sales['sqft_living'], 3)
poly3_data['price'] = sales['price']

model3 = linear_model.LinearRegression().fit(poly3_data.loc[:, poly3_data.columns != 'price'], np.array([poly3_data['price']]).T)
plt.plot(poly3_data['power_1'],poly3_data['price'],'.', poly3_data['power_1'], model3.predict(poly3_data.loc[:, poly3_data.columns != 'price']),'-')
plt.show()

poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
poly15_data['price'] = sales['price']

model15 = linear_model.LinearRegression().fit(poly15_data.loc[:, poly15_data.columns != 'price'], np.array([poly15_data['price']]).T)
plt.plot(poly15_data['power_1'], poly15_data['price'], '.', poly15_data['power_1'], model15.predict(poly15_data.loc[:, poly15_data.columns != 'price']), '-')
plt.show()

set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype = dtype_dict)
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype = dtype_dict)
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype = dtype_dict)
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype = dtype_dict)

def get_poly_model(set_data):
    poly15_data = polynomial_dataframe(set_data['sqft_living'], 15)
    poly15_data['price'] = sales['price']
    model15 = linear_model.LinearRegression().fit(poly15_data.loc[:, poly15_data.columns != 'price'], np.array([poly15_data['price']]).T)
    return poly15_data, model15

def get_coef(set_data):
    poly15_data, model15 = get_poly_model(set_data)
    return model15.coef_

def plot_fitted_line(set_data):
    poly15_data, model15 = get_poly_model(set_data)
    return plt.plot(poly15_data['power_1'], poly15_data['price'], '.', poly15_data['power_1'], model15.predict(poly15_data.loc[:, poly15_data.columns != 'price']), '-')

print(get_coef(set_1))
plot_fitted_line(set_1)
plt.show()

print(get_coef(set_2))
plot_fitted_line(set_2)
plt.show()

print(get_coef(set_3))
plot_fitted_line(set_3)
plt.show()

print(get_coef(set_4))
plot_fitted_line(set_4)
plt.show()

from sklearn.metrics import mean_squared_error
from scipy import sqrt

house_train = pd.read_csv('wk3_kc_house_train_data.csv', dtype = dtype_dict)
house_test = pd.read_csv('wk3_kc_house_test_data.csv', dtype = dtype_dict)
house_valid = pd.read_csv('wk3_kc_house_valid_data.csv', dtype = dtype_dict)

RSS_valid = 1e17
power_valid = 1
for power in range(1, 16):
    poly_train_data = polynomial_dataframe(house_train['sqft_living'], power)
    poly_valid_data = polynomial_dataframe(house_valid['sqft_living'], power)
    poly_train_data['price'] = house_train['price']
    poly_valid_data['price'] = house_valid['price']
    lin_reg_model = linear_model.LinearRegression().fit(poly_train_data.loc[:, poly_train_data.columns != 'price'],
                                                        np.array([poly_train_data['price']]).T)
    house_valid_predict = lin_reg_model.predict(poly_valid_data.loc[:, poly_valid_data.columns != 'price'])
    output_str = "Power " + str(power) + " RSS on VALIDATION Data: "
    output_RSS = sqrt(mean_squared_error(poly_valid_data['price'], house_valid_predict))
    print(output_str, output_RSS)
    if output_RSS < RSS_valid:
        RSS_valid = output_RSS
        power_valid = power

print("The 'best' polynomial degree is: ", power_valid)
print("The lowest RSS on VALIDATION data is: ", RSS_valid)