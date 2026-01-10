def greet():
    name = input("What's your name? \n");
    print(f"Hello, {name} ! Welcome to AI/ML engineering!")
    print("Let's build something amazing together.")

skill_to_learn = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "Langchain",
    "AI Engineering",
    "MLOps",
]

print("Skills I'm learing:")
for i, skill in enumerate(skill_to_learn, start =1):
    print(f"{i}. {skill}")

greet()