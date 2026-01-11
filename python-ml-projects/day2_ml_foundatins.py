"""
Day 2: Data Structures - The Foundation of ML
Every ML model works with data in these structures.
"""

#============================================================ 
#LISTS - Your ML Dataset Container
#============================================================ 


#This of this as a simple dataset 

tempratures = [72, 75, 78, 79, 80, 73, 77]

print("=" *50)
print("WORKING WITH LISTS(Datasets)")
print("=" *50)

# Accessing data
print(f"Day 1 temperature: {tempratures[0]}")
print(f"Last day temperature: {tempratures[-1]}")
print(f"First 3 days temperatures: {tempratures[:3]}")

# Statistics (foundation for ML)
avg_temp= sum(tempratures)/len(tempratures)
print(f"\nAverage temperature : {avg_temp: .2f} °F")

# Finding patterns
above_70 = [temp for temp in tempratures if temp > 70]
print(f"Days above 70°F: {above_70}")

#============================================================ 
# DICTIONARIES - Feature Engineering
#============================================================ 

print("\n" +"=" * 50)
print("DICTIONARIES (Data Records)")
print ("=" * 50)

person1 = {
    "age": 25,
    "income":50000,
    "education": "Bachelor",
    "credit_score":720
}

person2 = {
    "age": 35,
    "income":75000,
    "education": "Master",
    "credit_score":780
}

dataset = [person1, person2]
print("Dataset:")

for i, person in enumerate (dataset, 1):
    print(f"Person {i}: Age = {person['age']}, Income = {person['income']}, Education = {person['education']}, Credit Score = {person['credit_score']}")

#============================================================ 
# FUNCTIONS- Reusable ML Operations
#============================================================ 

print("\n" + "=" * 50)
print("FUNCTIONS (ML Operations)")
print("=" * 50)


def normalize_data(values):
    """
    Normalization - crucial ML processing step
    Scales data to 0-1 range 
    """

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    normalized = [(x - min_val) / range_val for x in values]
    return normalized
data = [10, 20, 30, 40, 50]
normalized = normalize_data(data)

print(f"Original data:{data}")
print(f"Normalized (0-1): {[f'{x:.2f}' for x in normalized]}")


def train_test_split(data, train_ratio = 0.8):
    """
    Split data into training and testing sets
    Core concept of ML
    """
    split_point = int (len(data) * train_ratio)
    train = data[:split_point]
    test = data[split_point:]
    return train, test
    
my_data = list(range(1, 11))  # Sample data from 1 to 11
train,test = train_test_split(my_data)
print(f"\nFull dataset : {my_data}")
print(f"Training set (80%): {train}")
print(f"Testing set (20%): {test}")

# ============================================
# LOOPS - Processing Datasets
# ============================================

print("\n" + "=" * 50)
print("FUNCTIONS (ML Operations)")
print("=" * 50)

# processing each data point
prices = [100, 150, 200, 175,225]
discounted_prices =[]

for price in prices:
    discount = price * 0.1
    new_price = price - discount
    discounted_prices.append(new_price)

print(f"Original prices: {prices}")
print(f"After 10% discount: {discounted_prices}")

#List comprehension (Pythonic way)
squared = [x**2 for x in range(1, 6)]
print(f"\nSquared numbers: {squared}")

# ============================================
# PRACTICAL EXERCISE
# ============================================
print("\n" + "=" * 50)
print("MINI PROJECT: Simple Prediction")
print("=" * 50)

# Simple linear relationship: y = 2x + 1

def simple_model(x):
    """Our first 'model' - just a linear function"""
    return 2 * x + 1

# Test our model
test_inputs = [1, 2, 3, 4, 5]
predictions = [simple_model(x) for x in test_inputs]

print("Input -> Prediction")
for x,y in zip(test_inputs,predictions):
    print(f"{x} -> {y}")

print("\n Day 2 complete! You've built a foundation for ML with data structures and functions.")











































