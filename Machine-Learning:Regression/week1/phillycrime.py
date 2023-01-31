# Load some house value vs. crime rate data
import turicreate as tc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sales = tc.SFrame.read_csv("Philadelphia_Crime_Rate_noNA.csv/")
# print(sales)

# Exploring the data
plt.scatter(data=sales, x="CrimeRate", y="HousePrice")
plt.show()
# input("Please enter to exit: ")