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