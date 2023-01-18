# Predicting houses prices
# Fire up graphlab create
import turicreate as tc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load some houses sales data
sales = tc.SFrame("home_data.sframe")
print(sales)


# Exploring the data for housing sales
sales.show()
tc.show(sales[1:5000]['sqft_living'], sales[1:5000]['price'])

# Create a simple regression model of sqft_living to price
train_data, test_data = sales.random_split(.8, seed=0)
sqft_model = tc.linear_regression.create(train_data, target="price", features=['sqft_living'])

# Evaluate the simple model
print(test_data['price'].mean())
print(sqft_model.evaluate(test_data))

# Let's show what our predictions look like
plt.plot(test_data["sqft_living"], test_data['price'], '.', test_data['sqft_living'], sqft_model.predict(test_data), '-')
plt.title("Predictions")
plt.show()
print(sqft_model.coefficients)

# Explore other features in the data
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
sales[my_features].show()

# Build a regression model with more features
my_features_model = tc.linear_regression.create(train_data, target='price', features=my_features)
print(my_features)
print(sqft_model.evaluate(test_data))
print(my_features_model.evaluate(test_data))

# Apply learned models to predict prices of 3 houses
house1 = sales[sales['id'] == '5309101200']
print(house1)
print(house1['price'])
print(sqft_model.predict(house1))
print(my_features_model.predict(house1))

# Prediction for a second, fancier house
house2 = sales[sales['id'] == '1925069082']
print(house2)

# Last house, super fancy
bill_gates = {'bedrooms':[8],
              'bathrooms':[25],
              'sqft_living':[50000],
              'sqft_lot':[225000],
              'floors':[4],
              'zipcode':['98039'],
              'condition':[10],
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]
              }

print(my_features_model.predict(tc.SFrame(bill_gates)))

#input("Press Enter to exit...")
