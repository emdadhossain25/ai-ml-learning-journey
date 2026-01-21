"""
Day 3: NumPy Practice Exercises
"""

import numpy as np

print("EXERCISE 1: Create a 5x5 array with random values")
# YOUR CODE HERE
arr =np.random.rand(5,5)
print(arr)

print("\nEXERCISE 2: Find the mean of each row")
# YOUR CODE HERE
row_means = np.mean(arr, axis =1) #axis 1 is for rows
print(row_means)

print("\nEXERCISE 3: Find the maximum value in each column")
# YOUR CODE HERE
col_maxes = np.max(arr, axis = 0) #axis 0 is for columns
print(col_maxes)

print("\nEXERCISE 4: Create array [10, 20, 30, ..., 100] without typing it")
# YOUR CODE HERE
arr2 = np.arange(10, 101, 10)
print(arr2)

print("\nEXERCISE 5: Find all values > 50 in the above array")
# YOUR CODE HERE
above_50 = arr2[arr2 > 50]
print(above_50)

print("\nEXERCISE 6: Create a 3x3 identity matrix")
# YOUR CODE HERE
identity = np.eye(3)
print(identity)

print("\nEXERCISE 7: Matrix multiplication")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# Multiply A and B
# YOUR CODE HERE
result =np.dot(A, B)
print(result)