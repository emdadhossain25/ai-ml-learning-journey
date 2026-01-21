# Day 2 Python Data Structures FOR ML
# Lists - will use these for datasets
numbers = [1,2,3,4,5]
print(f"My numbers : {numbers}")
print(f"Sum : {sum(numbers)}")
print(f"Average : {sum(numbers)/len(numbers)}")

# Dictionary - like database record 
persons = {
    "name": "ML Learner",
    "skills":["Python", "Git", "Determintation"],
    "day":2,
    "goal":"AI/ML Engineer"
}
print(f"\nName:{persons['name']}")
print(f"Current day:{persons['day']}")
print("Skills I am learning:")
for skill in persons['skills']:
    print(f"- {skill}")

# List operations -crucial for data manipulation
data = [10, 20, 30, 40, 50]
print(f"\n original data:{data}")
data.append(60)
print(f"After append: {data}")
print(f"First 3 items: {data[:3]}")
print(f"Last 2 items: {data[-2:]}")

# Function - building blocks of ML code

def calculate_stats(numbers_list):
    """Calculate basic statistics -  foundation for ML metrics"""
    return{
        "count": len(numbers_list),
        "sum": sum(numbers_list),
        "average": sum(numbers_list)/len(numbers_list),
        "min": min(numbers_list),
        "max": max(numbers_list)
    }

stats = calculate_stats([100,200,150,175,225])
print(f"\nStatistics: {stats}")
























