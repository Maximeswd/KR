import os
from collections import defaultdict
import matplotlib.pyplot as plt


def get_lines(filename):
    with open(filename) as f:
        return f.read().splitlines()


def get_numbers_per_row(lines):
    return [len(l.replace(".", "")) for l in lines]


files = os.listdir('./sudokus_files/')
for f in files:

    data = defaultdict(lambda: 0)
    for number in get_numbers_per_row(get_lines(f'./sudokus_files/{f}')):
        data[number] += 1

    nr_of_givens = list(data.keys())
    nr_of_sudokus = list(data.values())

    # creating the bar plot
    plt.bar(nr_of_givens, nr_of_sudokus, color='blue', width=0.4)

    plt.xlabel("Number of givens")
    plt.ylabel("Number of Sudoku's")
    plt.suptitle(f)

    # Show graphic
    plt.savefig(f'./{f}.png')
    plt.show()
