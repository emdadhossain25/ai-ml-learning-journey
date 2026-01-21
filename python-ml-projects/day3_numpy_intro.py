"""
Day 3: Numpy Fundamentals
Why Numpy is the foundation of ML
"""

import numpy as np
import time

print("=" * 60)
print("NUMPY VS PYTHON LISTS - THE SPEED DIFFERENCE") 
print("=" * 60)

#Python Lists approach

python_list = list(range(1000000))
start = time.time()

python_doubled = [x *2 for x in python_list]
python_time = time.time()-start


#Numpy array approach

numpy_array = np.arange(1000000)
start = time.time()
numpy_doubled = numpy_array * 2
numpy_time = time.time() - start

print(f"\n Python list time: {python_time:.6f} seconds")
print(f"\n Numpy Array time: {numpy_time:.6f} seconds")
print(f"\n Numpy is: {python_time/numpy_time:.2f}x FASTER!")

print("=" * 60)
print("CREATING ARRAYS")
print("=" * 60)

#Different ways to create arrays
arr1 = np.array([1,2,3,4,5])
print(f"\nFrom list: {arr1}")


arr2 = np.arange(0,10,2) #start, stop, step
print(f"Using arange: {arr2}")

arr3 = np.linspace(0,1,5) #start, stop, number of points
print(f"Using linspace: {arr3}")

arr4= np.zeros((5)) 
print(f"Array of zeros: {arr4}")

arr5 = np.ones(5)
print(f"Array of ones: {arr5}")

arr6 = np.random.rand(5) 
print(f"Random array: {arr6}")

print("=" * 60)
print("ARRAY PROPERTIES")
print("=" * 60)

data = np.array([10,20,30,40,50])
print(f"Array: {data}")
print(f"Shape: {data.shape}")
print(f"Data Type: {data.dtype}")
print(f"Data Dimensions: {data.ndim}")
print(f"Size (number of elements): {data.size}")

print("=" * 60)
print("ARRAY PROPERTIES")
print("=" * 60)

#vectorized operations (no loops needed!)
a = np.array([1,2,3,4,5])
b = np.array([10,20,30,40,50])

print(f"a = {a}")
print(f"b = {b}")
print(f"\n a+b = {a+b}")
print(f"\n a*b = {a*b}")
print(f"\n a-b = {a-b}")
print(f"\n a/b = {a/b}")
print(f"\n a**2 = {a**2}")


#operations with scalars
print(f"\n a * 10 = {a * 10}")
print(f"a+100 = {a + 100}")

print("=" * 60)
print("MATHEMATICAL FUNCTIONS")
print("=" * 60)

data = np.array([1,4,9,16,25])
print(f"Data: {data}")
print(f"Square Root: {np.sqrt(data)}")
print(f"sum: {np.sum(data)}")
print(f"Mean: {np.mean(data)}")
print(f"Standard Deviation: {np.std(data)}")
print(f"Max: {np.max(data)}")
print(f"Min: {np.min(data)}")

print("\n✅ NumPy basics complete!")
































