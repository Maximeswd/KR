import os
from collections import defaultdict

f = open("C:/Users/SZRis/Downloads/top2365.sdk.txt", "r") #use wherever you have the sudoku file
listy = []
green_cells = []
black_cells = []
for line in f:
    listy.append(line.strip()) #removes all newlines
for pip in listy: #to get the very green cells and very black cells of the pattern
    pattern_positions = (pip[0],pip[3],pip[27],pip[30],pip[33],pip[54],pip[57],pip[60])#excludes 6
    anti_pattern_positions = (pip[20], pip[23], pip[26], pip[47], pip[50], pip[53], pip[74], pip[77], pip[80])
    pattern = [value for value in pattern_positions if value != "."] #so if there is a number there, which is what we want
    anti_pattern = [value for value in anti_pattern_positions if value != "."]  # so if there is a number there, which is what we don't want
    #print (pattern)
    #print (len(pattern))
    if (len(pattern)) >= 2: #if there are two or more numbers in the green cells, we want them
        green_cells.append(pip)
        print (green_cells)
    if (len(anti_pattern)) <= 1:  # if there is or less number in the black cells, we want them
        black_cells.append(pip)
        print(black_cells)
        #now to get all the sudokus that satisfy both criteria back