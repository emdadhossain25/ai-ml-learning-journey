numbers = [1,2,3,4,5]
print(f"My numbers : {numbers}")
print(f"Sum : {sum(numbers)}")
print(f"Average : {sum(numbers)/len(numbers)}")

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

data = [10, 20, 30, 40, 50]
print(f"\n original data:{data}")
data.append(60)
print(f"After append: {data}")
print(f"First 3 items: {data[:3]}")
print(f"Last 2 items: {data[-2:]}")