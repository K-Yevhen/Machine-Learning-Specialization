# Create a dataset based on true sinusoidal relationship

import turicreate as tc
import math
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Create random values for x in internal [0,1)

random.seed(98103)
n = 30
x = tc.SArray([random.random() for i in range(n)]).sort()

# Compute y

y = x.apply(lambda x: math.sin(4*x))
