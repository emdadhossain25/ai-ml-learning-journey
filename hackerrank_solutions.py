"""
HackerRank Solutions - Day 2
"""

# Problem 1: Hello World
print("Hello, World!")

# Problem 2: If-Else (example)
n = 5;
if n % 2 !=0:
    print("Weird")
elif n % 2 == 0 and 2<=n <=5:
    print("Not Weird")
elif n % 2 == 0 and 6<=n <=20:
    print("Weird")
elif n % 2 == 0 and n>20:
    print("Not Weird")

# Problem 3: Arithmetic Operators
    a = 3
    b = 2
    print(f"{a+b}")
    print(f"{a-b}")
    print(f"{a*b}")


"""
HackerRank Solutions - Day 3
"""

# Problem Lists

N = int(input())
list = []
    
for _ in range(N):
    command = input().split()
    
    if command[0] == "insert":
        list.insert(int(command[1]), int(command[2]))
    elif command[0] == "print":
        print(list)
    elif command[0] == "remove":
        list.remove(int(command[1]))
    elif command[0] == "append":
        list.append(int(command[1]))
    elif command[0] == "sort":
        list.sort()
    elif command[0] == "pop":
        list.pop()
    elif command[0] == "reverse":
        list.reverse()


# Problem Tuples 

if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    t = tuple(integer_list)
    print (hash(t))


# problem Runner-Up Score
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arr = list(set(arr))
    arr.sort()
    print(arr[-2])