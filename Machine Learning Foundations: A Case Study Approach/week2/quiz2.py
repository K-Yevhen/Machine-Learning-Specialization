import turicreate as tc
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house
'grade', # measure of quality of construction
'waterfront', # waterfront property
'view', # type of view
'sqft_above', # square feet above ground
'sqft_basement', # square feet in basement
'yr_built', # the year built
'yr_renovated', # the year renovated
'lat', 'long', # the lat-long of the parcel
'sqft_living15', # average sq.ft. of 15 nearest neighbors
'sqft_lot15', # average lot size of 15 nearest neighbors
]

sales = tc.SFrame("home_data.sframe")
sales_pandas = pd.read_csv("home_data.csv")

# print(sales)
# print(sales_pandas)


# print(sales["zipcode"].show())
# print(sales["zipcode"])
# print(sales["zipcode" == '98178'])
high = sales[sales['zipcode' == '98178']]
# print(high["price"].mean())


in_range = sales[(sales['sqft_living'] >= 2000) & (sales['sqft_living'] <= 4000)]
#print(in_range)
# print(len(in_range))
# print(len(sales))
# print(9221//21613)

# print(sales[my_features].show())
# input("Press Enter to exit...")

train_data, test_data = sales.random_split(.8, seed=0)
sqft_model = tc.linear_regression.create(train_data, target="price", features=['sqft_living'])
my_features_model = tc.linear_regression.create(train_data, target='price', features=my_features)
# print(my_features)
# print(sqft_model.evaluate(test_data))
print(my_features_model.evaluate(test_data))

adv_features_model = tc.linear_regression.create(train_data,target='price',
                                                            features=advanced_features, validation_set=None)
print(adv_features_model.evaluate(test_data))
