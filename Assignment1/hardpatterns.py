#hard_pattern will hold the sudokus that satisfy the hard sudoku pattern

with open("C:/Users/SZRis/Downloads/AllSudokus.txt") as f: #paste your path here
    lines = f.read().splitlines()
lines = list(dict.fromkeys(lines)) #remove duplicates

green_cells = []
black_cells = []
dark_blue_cells = []
hard_pattern = []
more_than_17 = []
non_patterns = []
with_17 = []
completelist = []
for line in lines: #get the very green cells and very black cells of the pattern
    completelist.append(line)
    if len(line.replace(".", "")) == 17: #to get all the sudokus with 17 givens
        with_17.append(line)
print (completelist)
print (len(with_17))

for line in with_17:
    pattern_positions = (line[0], line[3], line[6], line[27], line[30], line[33], line[54], line[57], line[60])  #no longer excludes 6
    anti_pattern_positions = (line[20], line[23], line[26], line[47], line[50], line[53], line[74], line[77], line[80])
    dark_blue_positions = (line[11], line[14], line[17], line[19], line[22], line[25], line[38], line[41], line[44], line[46], line[49], line[52], line[65], line[68], line[71], line[73], line[76], line[79])
    pattern = [value for value in pattern_positions if value != "."]  # so if there is a number there, which is what we want
    anti_pattern = [value for value in anti_pattern_positions if value != "."]  # so if there is a number there, which is what we don't want
    dark_blue = [value for value in dark_blue_positions if value != "."] # so if there is a number there, which is what we don't want
    if (len(pattern)) >= 3:  # if there is this amount or more in the green cells, we want them
        green_cells.append(line)
    if (len(anti_pattern)) == 0:  # if there is this amount in the black cells, we want them
        black_cells.append(line)
    if (len(dark_blue)) <= 2:  # if there is this amount or less in the green cells, we want them
        dark_blue_cells.append(line)
    hard_pattern = list(set(green_cells) & set(black_cells) & set(dark_blue_cells)) #get the sudokus that are satisfy both criteria
    non_patterns = [x for x in with_17 if x not in hard_pattern]
#print (len(less_than_22_more_than_17))
#print (len(non_patterns)) #all the others
print (hard_pattern) #the ones with the pattern
print (len(hard_pattern)) #how many we have, just to visualize
print (len(non_patterns))
both = set(completelist).intersection(non_patterns) #this is for the nonpatterned ones, but whichever one you need
indices = [completelist.index(x) for x in both] #to get the index numbers of the fitting sudokus
print(indices)

