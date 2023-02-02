# Load some house value vs. crime rate data
import turicreate as tc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sales = tc.SFrame.read_csv("Philadelphia_Crime_Rate_noNA.csv/")
# print(sales)

# Exploring the data
# plt.scatter(data=sales, x="CrimeRate", y="HousePrice")
# plt.show()
# input("Please enter to exit: ")

# Fit the regression model using crime as the features
crime_model = tc.linear_regression.create(sales, target='HousePrice', features=['CrimeRate'], validation_set=None, verbose=False)

# Visualisation
# plt.plot(sales['CrimeRate'], sales['HousePrice'], '.',
#          sales['CrimeRate'], crime_model.predict(sales), '-')
# plt.show()

# Remove Center City and redo the analysis
sales_noCC = sales[sales['MilesPhila'] != 0.0]
# plt.scatter(data=sales_noCC, x="CrimeRate", y="HousePrice")
# plt.show()

# Refit our simple regression model on this modified dataset:
crime_model_noCC = tc.linear_regression.create(sales_noCC, target="HousePrice", features=['CrimeRate'], validation_set=None, verbose=False)
# plt.plot(sales_noCC["CrimeRate"], sales_noCC['HousePrice'], '.',
#          sales_noCC["CrimeRate"], crime_model.predict(sales_noCC), '-')
# plt.show()

# Compare coefficients for full-data fit versus no-Center-City fit
print(crime_model.coefficients)
