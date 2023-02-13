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
plt.plot(poly15_data['power_1'], poly15_data['price'],'.', poly15_data['power_1'], model15.predict(poly15_data.loc[:, poly15_data.columns != 'price']), '-')


set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype = dtype_dict)
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype = dtype_dict)
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype = dtype_dict)
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype = dtype_dict)
