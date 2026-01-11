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