with open("./sudokus_files/top2365.sdk.txt") as f:
    lines = f.read().splitlines()

less_than_22 = []

for line in lines:
    if len(line.replace(".", "")) <= 22:
        less_than_22.append(line)

print(len(less_than_22))
