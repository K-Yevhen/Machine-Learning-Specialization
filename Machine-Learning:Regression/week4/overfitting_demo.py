# Create a dataset based on true sinusoidal relationship

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
    plt.plot(data['X1'],data['Y'],'k.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# plot_data(data)
