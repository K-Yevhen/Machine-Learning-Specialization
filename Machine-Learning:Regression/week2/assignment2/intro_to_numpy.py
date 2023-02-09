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

# 2D arrays are accessed similarly by referring to the row and column index separated by a comma:
my_matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(my_matrix)
print(my_matrix[1, 2])

# Sequences of indices can be accessed using ':' for example
print(my_matrix[0:2, 1]) # recall 0:2 = [0, 1]
print(my_matrix[0, 0:3])

# You can also pass a list of indices.
fib_indices = np.array([1, 1, 2, 3])
random_vector = np.random.random(10) # 10 random numbers between 0 and 1
print(random_vector)
print(random_vector[fib_indices])

# You can also use true/false values to select values
my_vector = np.array([1, 2, 3, 4])
select_index = np.array([True, False, True, False])
print(my_vector[select_index])

# For 2D arrays you can select specific columns and specific rows. Passing ':' selects all rows/columns
select_cols = np.array([True, False, True])  # 1st and 3rd column
select_rows = np.array([False, True])  # 2nd row
print my_matrix[select_rows, :]  # just 2nd row but all columns
print my_matrix[:, select_cols]  # all rows and just the 1st and 3rd column

# Operations on Arrays
my_array = np.array([1., 2., 3., 4.])
print(my_array * my_array)
print(my_array**2)
print(my_array - np.ones(4))
print(my_array + np.ones(4))
print(my_array / 3)
print(my_array / np.array([2., 3., 4., 5.]))  # = [1.0/2.0, 2.0/3.0, 3.0/4.0, 4.0/5.0]

# You can compute the sum with np.sum() and the average with np.average()
print(np.sum(my_array))
print(np.average(my_array))
print(np.sum(my_array)/len(my_array))

# The dot product
# An important mathematical operation in linear algebra is the dot product.
# When we compute the dot product between two vectors we are simply multiplying them elementwise and adding them up. In numpy you can do this with np.dot()
array1 = np.array([1., 2., 3., 4.])
array2 = np.array([2., 3., 4., 5.])
print(np.dot(array1, array2))
print(np.sum(array1*array2))

array1_mag = np.sqrt(np.dot(array1, array1))
print(array1_mag)
print(np.sqrt(np.sum(array1*array1)))

my_features = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
print(my_features)

my_weights = np.array([0.4, 0.5])
print(my_weights)

my_predictions = np.dot(my_features, my_weights)  # note that the weights are on the right
print(my_predictions)  # which has 4 elements since my_features has 4 rows

my_matrix = my_features
my_array = np.array([0.3, 0.4, 0.5, 0.6])

print(np.dot(my_array, my_matrix))  # which has 2 elements because my_matrix has 2 columns

# Multiplying Matrices
matrix_1 = np.array([[1., 2., 3.], [4., 5., 6.]])
print(matrix_1)

matrix_2 = np.array([[1., 2.], [3., 4.], [5., 6.]])
print(matrix_2)

print(np.dot(matrix_1, matrix_2))
