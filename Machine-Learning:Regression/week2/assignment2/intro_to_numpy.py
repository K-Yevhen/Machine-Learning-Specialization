import numpy as np

# Creating Numpy Arrays
mylist = [1., 2., 3., 4.]
mynparray = np.array(mylist)
print(mynparray)

one_vector = np.ones(4)
print(one_vector)  # using print removes the array() portion

one2Darray = np.ones((2, 4))  # an 2D array with 2 "rows" and 4 "columns"
print(one2Darray)

empty_vector = np.empty(5)
print(empty_vector)

# Accessing array elements
print(mynparray[2])

my_matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(my_matrix)
print(my_matrix[1, 2])
print(my_matrix[0:2, 1]) # recall 0:2 = [0, 1]
print(my_matrix[0, 0:3])

fib_indices = np.array([1, 1, 2, 3])
random_vector = np.random.random(10) # 10 random numbers between 0 and 1
print random_vector