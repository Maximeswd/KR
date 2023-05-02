# Packages
import pandas as pd
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats
import seaborn as sns
import itertools

# convert string into matrix to represent input of the puzzle
def generate_matrix(string):
    matrix = [[0 for x in range(9)] for x in range(9)]
    for i in range(9):
        for j in range(9):
            matrix[i][j] = string[i * 9 + j]
    return matrix

# Create CNF files in this directory for each sudoku
def create_cnf_file(sudoku, index):
    output_file = open('sudoku' + str(index) + '.cnf', 'w')
    output_file.write("p cnf 999 999999" + "\n")
    generate_clauses_for_filled_cells(output_file, sudoku)
    generate_clauses_for_each_cell(output_file)
    generate_clauses_for_each_column(output_file)
    generate_clauses_for_each_row(output_file)
    generate_clauses_for_sub_matrix(output_file)
    output_file.close()

# Create clauses per filled cell
def generate_clauses_for_filled_cells(output_file, sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] != '0':
                output_file.write(str(code(i + 1, j + 1, int(sudoku[i][j]))) + ' 0\n')

# Create clauses for each cell
def generate_clauses_for_each_cell(output_file):
    for i in range(1, 10):
        for j in range(1, 10):
            for d in range(1, 10):
                output_file.write(str(code(i, j, d)) + ' ')
            output_file.write(' 0\n')

# Create clauses for each column
def generate_clauses_for_each_column(output_file):
    for i in range(1, 10):
        for d in range(1, 10):
            for j in range(1, 9):
                for index in range(j + 1, 10):
                    output_file.write("-" + str(code(i, j, d)) + " -" + str(code(i, index, d)) + " 0\n")

# Create clauses for each row
def generate_clauses_for_each_row(output_file):
    for j in range(1, 10):
        for d in range(1, 10):
            for i in range(1, 9):
                for index in range(i + 1, 10):
                    output_file.write("-" + str(code(i, j, d)) + " -" + str(code(index, j, d)) + " 0\n")

# Create clauses for the created sub matrix
def generate_clauses_for_sub_matrix(output_file):
    for d in range(1, 10):
        for x_axis in range(0, 3):
            for y_axis in range(0, 3):
                for i in range(1, 4):
                    for j in range(1, 4):
                        for k in range(j + 1, 4):
                            index_1 = x_axis * 3 + i
                            index_2 = y_axis * 3 + j
                            index_3 = y_axis * 3 + k
                            output_file.write('-' + str(code(index_1, index_2, d)) + ' -' + str(code(index_1, index_3, d)) + ' 0\n')

                        for k in range(i + 1, 4):
                            for l in range(1, 4):
                                index_1 = x_axis * 3 + i
                                index_2 = y_axis * 3 + j
                                index_3 = x_axis * 3 + k
                                index_4 = y_axis * 3 + l
                                output_file.write('-' + str(code(index_1, index_2, d)) + ' -' + str(code(index_3, index_4, d)) + ' 0\n')

def code(i, j, d):
    return 100*i+10*j+d

# Read in the text files of all the sudokus
f = open("/Users/gast/Desktop/VS code/Projects/AllSudokus.txt", 'r')

with open("/Users/gast/Desktop/VS code/Projects/AllSudokus.txt") as f:
    lines = []
    while True:
        line = f.readline()
        if not line:
            break
        lines.append(line.strip())

# Remove duplicates
def getUniqueItems(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Change the dots to zero's 
sudokus = [w.replace('.', '0') for w in getUniqueItems(lines)]

# Convert all the files to CNF and save files 
if __name__ == "__main__":
    input_filenames = sudokus
    index = 1
    for input_file in input_filenames:
        input_matrix = generate_matrix(input_file)
        create_cnf_file(input_matrix, index)
        index += 1

### Definition hard sudokus 1: Number of givens

# Get counts of givens
def get_count_givens(sudoku):
    list_of_ints = [int(x) for x in sudoku]
    count = np.count_nonzero([int(x) for x in list_of_ints])
    return count

n_givens = []
for i in sudokus:
    n_givens.append(get_count_givens(i))

# Get all the indexes of the sudokus with the same number of givens 
def get_index(l):
    _indices = defaultdict(list)

    for index, item in enumerate(l):
        _indices[item].append(index+1)
    
    return list(_indices.items())

# Get a sorted dataframe of all the sudoku indexes per nr. of givens
d_sorted = sorted(get_index(n_givens), key=lambda tup: tup[0])

# Count the number of sudokus per nr. of givens 
def freq_info_givens(d):
    givens = []
    givens_count = []
    for i in range(0, len(d)):
        givens.append(d[i][0])
        givens_count.append(len(d[i][1]))
    df_counts = pd.DataFrame({'N givens' : givens, 'Counts' : givens_count})
    return df_counts

# Plot the frequencies
df_counts = freq_info_givens(d_sorted)
df_counts.plot(kind='bar', x = 'N givens', y = 'Counts')
plt.show()

#..................... Retrieve the Specific Sudoku Pattern .....................#

# Only use the sudokus with 17 givens
sudokus17 = [sudokus[i] for i in [x-1 for x in d_sorted[0][1]]]

# Retrieve the hard pattern
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
print(hard_pattern) #the ones with the pattern
print(len(hard_pattern)) #how many we have, just to visualize
print(len(non_patterns))
both = set(completelist).intersection(non_patterns) #this is for the nonpatterned ones, but whichever one you need
indices = [completelist.index(x) for x in both] #to get the index numbers of the fitting sudokus

# Retrieve the sudokus that contain the pattern or not
def partition_on_index(it, ind):
    indices = set(ind)   
    l1, l2 = [], []
    l_append = (l1.append, l2.append)
    for idx, element in enumerate(it):
        l_append[idx in ind](element)
    return l1, l2
withoutP, withP = partition_on_index(sudokus17, indices)

# Retrieve the sudoku numbers per group (pattern vs. no pattern)
def find_index(bigL, smallL):
    a = [[i for i in range(len(bigL)) if item1 == bigL[i]] for item1 in smallL]
    return [x+1 for x in list(itertools.chain.from_iterable(a))]
pattern = find_index(sudokus, withP)
nopattern = find_index(sudokus, withoutP)

#..................... Conflict Driven Clause Learning .....................#

## Heuristics
# 1. Value Selection Heuristic to minimize search space: Two - Literal Watch data structure based BCP
# 2. Variable Selection Decision heuristic : Variable State Independent Decaying Sum (VSIDS)
# 3. Restart strategy: Random Restarts with a decaying probability

# Function to read the clauses from the cnf files 
def read_clause (fname):
    f = open(fname,"r")
    read_count = 0
    clauses = []
    alert = 1                                       # to indicate new clause, where a clause could extend beyond a single line of text 
    for x in f :                                    # for each line
        if x[0] != 'c' :                            # if not comment
            if x[0] == 'p' :                        # this is the problem defintion line
                x = x.split()
                num_var = int(x[2])                 # To read the nr. of variables and nr. of clauses
                num_claus = int(x[3])
            else :                          
                if alert == 1 :                     # Start a new clause 
                    newclause = []
                x = x.split()
                for j in x :
                    i = int(j)
                    alert = 0
                    if i == 0 :                     # If the clause ends, push to a list of clauses & start new clause
                        clauses.append(newclause)
                        alert = 1
                        read_count += 1
                        break
                    if -i in newclause:
                        alert = 1
                        read_count += 1
                        break
                    if i not in newclause:
                        newclause.append(i)        #  If the int is non-zero, append to the current clause
                        
                    # These conditions are to check for inconsistent/invalid cnf input 
                    if i > num_var :
                        print("Illegal variable number "+str(i))
                        return 0,num_var,num_claus
                    if read_count >= num_claus :
                        print("Nr. of clauses exceeds definition")
                        return 0,num_var,num_claus
    return 1,num_var,num_claus,clauses

## Heuristic 1. Value Selection Heuristic to minimize search space: Two - Literal Watch data structure based BCP

# BCP & Unit Propogation is ONLY used in the beginning to get rid of the unit clauses and their implications
def bcp(clauses, literal):                        # BCP on Literal
    new_claus_set = [x[:] for x in clauses]       
    for x in reversed(new_claus_set):
        if literal in x:                          # If the clause is satified,
            new_claus_set.remove(x)                    # Remove that specific clause
        if -literal in x:                         # If -literal is present, remaining should satisfy. Hence,
            x.remove(-literal)                         # Remove -literal from that specific clause
            if not x:                             # If this makes a clause empty, return UNSAT
                return -1
    return new_claus_set

def unit_propagation(clauses):                    # Propogate the unit clauses and add implications to M
    assignment = []
    flag=1
    while flag!=0:                                # Only stop when implications are found
        flag=0
        for x in clauses:                         
            if len(x) == 1 :                      # if the current clause is a UNIT clause; propagate + add to assignment
                unit=x[0]
                clauses = bcp(clauses, unit) 
                assignment += [unit]
                flag=1
            if clauses == -1:                     # Return -1 if UNSAT after propogating
                return -1, []
            if not clauses:                   
                return clauses, assignment
    return clauses, assignment

## Heuristic 2. Variable Selection Decision heuristic : Variable State Independent Decaying Sum (VSIDS)

# Initiatilaztion: Count the number of times a literal occurs
def VSIDS_init(clauses,num_var):
    counter = {}
    for x in range(-num_var,num_var+1):
        counter[x]=0
    for clause in clauses:
        for literal in clause:
            counter[literal] += 1
    return counter

# Conflict: Count increments of literals in conflict clause to increase chances of getting selected
def VSIDS_conflict(counter,conflictClause):
    for literal in conflictClause:
        counter[literal]+=1
    return counter

# Decay: Count for all literals at each conflict is reduced by 5%
def VSIDS_decay(counter,num_var):
    for i in range(-num_var,num_var+1):
        counter[i]=counter[i]*95/100
    return counter

# Decide: Selects a variable which is NOT yet in M based on the maximum counter value
def VSIDS_decide(counter,M,num_var):
    max=0
    var=0
    for x in range(-num_var,num_var+1):
        if counter[x]>=max and x not in M and -x not in M:
                max=counter[x]
                var=x
    return var

## Heuristic 3. Restart strategy: Random Restarts with a decaying probability

# Function for the CDCL solver with all other functions incuded, incl. the restart strategy
def CDCL_solve(clauses,num_var):
    decide_pos = []                                                 # To maintain the decision level
    M = []                                                          # Current assignments and implications
    clauses,M = unit_propagation(clauses)                           # Initial unit propogation: if there is a conflict -> UNSAT
    if clauses == -1 :
        return -1,0,0,0,0                                           # UNSAT
    back=M[:]                                                       # Keep initialization backup for the Restart strategy
    counter = VSIDS_init(clauses,num_var)                           # Initialize the heuristic counter
    
    # Initialize TWO LITERAL WATCH data Structure :
    literal_watch,clauses_literal_watched = create_watchList(clauses,M,num_var)

    probability=0.9                                                 # Random restart probability; for decays with restarts
    Restart_count = Learned_count = Decide_count = Imp_count = 0
    
    while not all_vars_assigned(num_var , len(M)) :                 # While variables remain to assign
        variable = VSIDS_decide(counter,M,num_var)                      # Decide : Pick a variable
        Decide_count += 1
        progressBar(len(M),num_var)                                 # print a designed progress bar   
        assign(variable,M,decide_pos)
        conflict,literal_watch = two_watch_propogate(clauses,literal_watch,clauses_literal_watched,M,variable)    # Deduce by Unit Propogation
        
        
        while conflict != -1 :
            VSIDS_conflict(counter,conflict)                                                                     # Increments counter of literals that are in conflict
            counter=VSIDS_decay(counter,num_var)                                                                 # Decay counters by 5%

            Learned_c = Analyze_Conflict(M, conflict,decide_pos)                                                 # Diagnose conflict

            dec_level = add_learned_clause_to(clauses,literal_watch,clauses_literal_watched,Learned_c,M)         # add the learned clause to all data structures
            Learned_count += 1
            jump_status,var,Imp_count = Backjump(M, dec_level, decide_pos,Imp_count)                             # BackJump to decision level

            if jump_status == -1:                                                                                # -> UNSAT
                return -1,Restart_count,Learned_count,Decide_count,Imp_count
            M.append(var)                                                                                        # Append negation of last literal after backjump
            
            probability,Restart_count = RandomRestart(M,back,decide_pos,probability,Restart_count)               # Random Restart
            conflict,literal_watch = two_watch_propogate(clauses,literal_watch,clauses_literal_watched,M,var)

            
    # If all variables are assigned; return  
    return M,Restart_count,Learned_count,Decide_count,Imp_count
    

# Function to create the two-literal watch data structure
def create_watchList(clauses,M,num_var):            
    literal_watch = {}                              # Will contain the main Literal -> Clause number mapping
    clauses_literal_watched = []                    # The reverse,i.e. Clause -> Literal mapping
    for i in range (-num_var,num_var+1):
        literal_watch[i] = []
    for i in range (0,len(clauses)):                # Pick two literals for each clause 
        newc = []
        first = 0
        for j in range(0,len(clauses[i])):
            if clauses[i][j] not in M and first==0:
                A = clauses[i][j]
                first=1
                continue
            if clauses[i][j] not in M and first==1:
                B = clauses[i][j]
                break
        newc.append(A)
        newc.append(B)
        clauses_literal_watched.append(newc)       # Add both to clause to watch of those literals
        literal_watch[A].append(i)                 
        literal_watch[B].append(i)
    return literal_watch,clauses_literal_watched


# Function to propogate using the two-literal watch
def two_watch_propogate(clauses,literal_watch,clauses_literal_watched,M,variable): 
    prop_list = [variable]                                              # Add current change to list of updates
    while len(prop_list) != 0 :                                         # While updates remain to propogate
        variable = prop_list.pop()                                      # Select one variable
        for affected_claus_num in reversed(literal_watch[-variable]) :    
            affected_claus = clauses[affected_claus_num][:]
            A = clauses_literal_watched[affected_claus_num][0]
            B = clauses_literal_watched[affected_claus_num][1]
            A_prev=A
            B_prev=B
            status,M,A,B,unit = check_status(affected_claus,M,A,B)      # Check status of each clause
            if status == "Unit" :
                prop_list.append(unit)
                M.append(unit)                                          # If unit clause, add to updates
            elif status == "Unsatisfied":                               # If -> UNSAT, return conflict clause
                return affected_claus,literal_watch
                                                                        # else the clause is still unresolve; remove from current & add to watcher
            literal_watch [A_prev].remove(affected_claus_num)
            literal_watch [B_prev].remove(affected_claus_num)
            clauses_literal_watched[affected_claus_num][0] = A
            clauses_literal_watched[affected_claus_num][1] = B
            literal_watch [A].append(affected_claus_num)
            literal_watch [B].append(affected_claus_num)
            
    return -1,literal_watch

# Function to check the status of literal watch
def check_status(clause,M,A,B):
    unit = 0
    if A in M or B in M:                                                # if one watch satisfied, nothing to do 
        return "Satisied",M,A,B,unit
    sym=[]                                                              # symbols not defined yet
    for literal in clause:                                              # find symbols not defined
        if -literal not in M:
            sym.append(literal)
        if literal in M :
            if -A not in M :
                return "Satisied",M,A,literal,unit
            return "Satisied",M,literal,B,unit
    if len(sym) == 1:                                                   # -> Unit Clause
        return "Unit",M,A,B,sym[0]
    if len(sym) == 0:                                                   # -> Unsatisfied conflict clause
        return "Unsatisfied",M,A,B,unit
    else :
        return "Unresolved",M,sym[0],sym[1],unit                        # Else return two new unsatisfied variables to use for Literal_watch

# Function to random restart
def RandomRestart(M,back,decide_pos,probability,Restart_count):  
    if random.random() < probability :                                  # If Generated random probability less than current : RESTART
        M = back[:]
        decide_pos = []
        probability *= 0.5                                              # Decay next Restart probability by 50%
        Restart_count += 1
        if probability < 0.001 :
            probability = 0.2
        if Restart_count > len(M) + 10:                                 # Avoid restarts if already restarted many times
            probability=0
    return probability,Restart_count

# Function to verify the Solution in M for SAT
def verify(M,clauses) :                                                 
    for c in clauses :                                                  
        flag = 0
        for lit in c:
            if lit in M:                                                # At least one literal should be true
                flag = 1
                break
        if flag == 0:
            return False
    return True

# Function to make sure that all decisions that are made untill now are a learned clause 
def Analyze_Conflict(M, conflict,decide_pos):       
    learn = []
    for x in decide_pos:
        learn.append(-M[x])
    return learn

# Function that checks if all variables are already assigned or not
def all_vars_assigned(num_var ,M_len):            
    if M_len >= num_var:
        return True
    return False

# Function to adds the decision literal to M & adds the correponding update to decision level 
def assign(variable,M,decide_pos):                
    decide_pos.append(len(M))
    M.append(variable)

# Function that adds the learned clause as a decicion if its a unit clause, and as a learned clause otherwise
def add_learned_clause_to(clauses,literal_watch,clauses_literal_watched,Learned_c,M):
    if len(Learned_c) == 0:
        return -1
    if len(Learned_c) == 1:                         # If unit clause is learned: add the clause as a decision 
        M.append(Learned_c[0])
        return 1,Learned_c[0]
    clauses.append(Learned_c)                       # For other clauses; add two literals (A, B) to the literal watch data structure
    A = Learned_c[0]
    B = Learned_c[1]
    i = len(clauses)-1
    newc = []
    newc.append(A)
    newc.append(B)
    clauses_literal_watched.append(newc)
    literal_watch[A].append(i)
    literal_watch[B].append(i)
    return 0

# Function to BackJump to decision level by deleting decisions from M and decision positions
def Backjump(M, dec_level, decide_pos,Imp_count):    
    Imp_count = Imp_count + len(M) - len(decide_pos)
    if not decide_pos:
        return -1,-1,Imp_count
    dec_level = decide_pos.pop()
    literal = M[dec_level]
    del M[dec_level:]
    return 0,-literal,Imp_count

# Function that prints a progress bar
def progressBar(current, total, barLength = 20) :    
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress (num_var:may backtrack): [%s%s] %d ' % (arrow, spaces, current), end='\r')

# Function to solve per sudoku
def solve(number):
    
    # Read the input file
    a,num_var,num_claus,clauses = read_clause("sudoku%s.cnf"%(number))                  # Read from input file
    
    # Print if the file could be read 
    if a == 1:                                                       # Status of reading input
        print("Successfully read the file")
    else:
        print("Could not read the file")
        return

    startSolve = time.process_time()
    solution = CDCL_solve(clauses, num_var)                          # Solve CNF by CDCL
    EndSolve = time.process_time()

    if solution[0] != -1:

        result_dict = {
            "Satisfied": "SAT", 
            "Restarts": solution[1],
            "Learned Clauses": solution[2],
            "Unit Clauses": solution[3],
            "Implications": solution[4], 
            "Solve Time Sec": EndSolve-startSolve, 
            }
    else :
        result_dict = {
            "Satisfied": "UNSAT", 
            "Restarts": solution[1],
            "Learned Clauses": solution[2],
            "Unit Clauses": solution[3],
            "Implications": solution[4], 
            "Solve Time Sec": EndSolve-startSolve, 
            }
    
    return result_dict

# Function to get the results for the same N for each group
def get_results(lis, n):
    results = []
    inds = random.sample(lis, n)
    for i in inds:
        results.append(solve(i))
    return results

# Get the information of all the solved sudokus per group (pattern vs no pattern)
smallest_group = np.min([len(pattern), len(nopattern)])
res_pattern = get_results(pattern, smallest_group)
res_nopattern = get_results(nopattern, smallest_group)


#### COMPARISON PATTERN vs. NO PATTERN ####

# Put the results of CDCL into a dataframe
CDCLpattern = pd.DataFrame(res_pattern)
CDCLnopattern = pd.DataFrame(res_nopattern)

# Put the results of DPLL into a dataframe
DPLLpattern = pd.read_csv("results_17_pattern.csv")
DPLLnopattern = pd.read_csv("results_17_no_pattern.csv")

# Shapiro wilk test (significant p-value = not normally distributed)
stats.shapiro(DPLLpattern['Solve Time Sec'])    # ShapiroResult(statistic=0.8540375232696533, pvalue=6.763571036572102e-07)
stats.shapiro(DPLLnopattern['Solve Time Sec'])  # ShapiroResult(statistic=0.9668149352073669, pvalue=0.05418872460722923)
stats.shapiro(CDCLpattern['Solve Time Sec'])    # ShapiroResult(statistic=0.5392369031906128, pvalue=1.103192922213489e-13)
stats.shapiro(CDCLnopattern['Solve Time Sec'])  # ShapiroResult(statistic=0.599725604057312, pvalue=1.0391762320441367e-12)

# Density plots --> To check if the data is normally distributed
sns.displot(DPLLpattern, x="Solve Time Sec", kind="kde")
sns.displot(DPLLnopattern, x="Solve Time Sec", kind="kde")
sns.displot(CDCLpattern, x="Solve Time Sec", kind="kde")
sns.displot(CDCLnopattern, x="Solve Time Sec", kind="kde")
plt.show()

# Check the Q-Q plots --> Also to check if the data is normally distributed
stats.probplot(DPLLpattern['Solve Time Sec'], dist="norm", plot=plt)
stats.probplot(DPLLnopattern['Solve Time Sec'], dist="norm", plot=plt)
stats.probplot(CDCLpattern['Solve Time Sec'], dist="norm", plot=plt)
stats.probplot(CDCLnopattern['Solve Time Sec'], dist="norm", plot=plt)
plt.show()

# Median values for each
np.median(DPLLpattern['Solve Time Sec'])    # 42.8046875
np.median(DPLLnopattern['Solve Time Sec'])  # 41.4140625
np.median(CDCLpattern['Solve Time Sec'])    # 6.776032000000214
np.median(CDCLnopattern['Solve Time Sec'])  # 8.761968000000707

# First check if the variances are equal or not

# Solve time
np.var(DPLLpattern['Solve Time Sec']), np.var(DPLLnopattern['Solve Time Sec'])      # (95.16162787543405, 59.683690765757625)
np.var(CDCLpattern['Solve Time Sec']), np.var(CDCLnopattern['Solve Time Sec'])      # (125.4278593557862, 211.1545814126803)

# Number of clauses DPLL and learned clauses CDCL
np.var(DPLLpattern['Learned Clauses']), np.var(DPLLnopattern['Learned Clauses'])    # (3206.222029320988, 2713.2035108024697) 
np.var(CDCLpattern['Learned Clauses']), np.var(CDCLnopattern['Learned Clauses'])    # (10861432.854166666, 14205909.024691356)

# Unit Clauses
np.var(DPLLpattern['Decisions']), np.var(DPLLnopattern['Decisions'])            # (4977.515432098766, 4274.101658950617)
np.var(CDCLpattern['Unit Clauses']), np.var(CDCLnopattern['Unit Clauses'])      # (10852412.081597222, 14204301.569251543)

# Restarts for CDCL
np.var(CDCLpattern['Restarts']), np.var(CDCLnopattern['Restarts'])  # (366.04924945184683, 173.4397031539888)

# Conduct Welch's T-Test 
stats.ttest_ind(DPLLpattern['Solve Time Sec'], DPLLnopattern['Solve Time Sec'], equal_var = False)   # Ttest_indResult(statistic=0.920784615435312, pvalue=0.35880638338535387)
stats.ttest_ind(CDCLpattern['Solve Time Sec'], CDCLnopattern['Solve Time Sec'], equal_var = False)   # Ttest_indResult(statistic=-1.465172533072887, pvalue=0.14522766906911092)
stats.ttest_ind(DPLLpattern['Learned Clauses'], DPLLnopattern['Learned Clauses'], equal_var = False) # Ttest_indResult(statistic=-0.8274767435032739, pvalue=0.40936495381558635)
stats.ttest_ind(CDCLpattern['Learned Clauses'], CDCLnopattern['Learned Clauses'], equal_var = False) # Ttest_indResult(statistic=-1.4756328853711522, pvalue=0.14229577519145234)
stats.ttest_ind(DPLLpattern['Unit Clauses'], DPLLnopattern['Unit Clauses'], equal_var = False)       # Ttest_indResult(statistic=-0.23959616007189308, pvalue=0.8109672225952245)
stats.ttest_ind(CDCLpattern['Unit Clauses'], CDCLnopattern['Unit Clauses'], equal_var = False)       # Ttest_indResult(statistic=0.7321111224475626, pvalue=0.46547883755617714)

#### COMPARISON DPLL vs. CDCL ####

# Although our research did not study the difference between the DPLL and CDCL algorithm, because the CDCL
# is built upon the DPLL to correct for the shortcomings of DPLL. This is confirmed because CDCL is significantly
# faster than DPLL
# The second hypothesis is that CDCL outperforms DPLL. 
np.var(DPLLpattern['Solve Time Sec']), np.var(CDCLpattern['Solve Time Sec'])            # (126.29345110170877, 117.81458393304908)
np.var(DPLLnopattern['Solve Time Sec']), np.var(CDCLnopattern['Solve Time Sec'])        # (175.6133970407636, 57.29805233657496)
np.var(DPLLpattern['Learned Clauses']), np.var(CDCLpattern['Learned Clauses'])          # (30642.400067465005, 8511719.874177769)
np.var(DPLLnopattern['Learned Clauses']), np.var(CDCLnopattern['Learned Clauses'])      # (25914.36464833868, 3019903.7082138644)
np.var(DPLLpattern['Unit Clauses']), np.var(CDCLpattern['Unit Clauses'])                # (54630.34879406307, 8625748.86591331)
np.var(DPLLnopattern['Unit Clauses']), np.var(CDCLnopattern['Unit Clauses'])            # (48619.69876876371, 3093928.6716140998)


stats.ttest_ind(DPLLpattern['Solve Time Sec'], CDCLpattern['Solve Time Sec'], equal_var = False)        # Ttest_indResult(statistic=19.139293178972583, pvalue=7.831898814355666e-41)
stats.ttest_ind(DPLLnopattern['Solve Time Sec'], CDCLnopattern['Solve Time Sec'], equal_var = False)    # Ttest_indResult(statistic=14.94324165245694, pvalue=4.617754499263488e-28)
stats.ttest_ind(DPLLpattern['Learned Clauses'], CDCLpattern['Learned Clauses'], equal_var = False)      # Ttest_indResult(statistic=-3.773147590379902, pvalue=0.00033097758697309853)
stats.ttest_ind(DPLLnopattern['Learned Clauses'], CDCLnopattern['Learned Clauses'], equal_var = False)  # Ttest_indResult(statistic=-5.242515983592029, pvalue=1.5580542964137023e-06)
stats.ttest_ind(DPLLpattern['Decisions'], CDCLpattern['Unit Clauses'], equal_var = False)               # Ttest_indResult(statistic=-2.913053339735594, pvalue=0.004780671824107488)
stats.ttest_ind(DPLLnopattern['Decisions'], CDCLnopattern['Unit Clauses'], equal_var = False)           # Ttest_indResult(statistic=-4.493512655393391, pvalue=2.6560797197433788e-05)


######## VIOLIN #########

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Algorithm')

# create test data
data1 = [DPLLpattern['Solve Time Sec'].tolist(), CDCLpattern['Solve Time Sec'].tolist()]        # pattern sudokus
data2 = [DPLLnopattern['Solve Time Sec'].tolist(), CDCLnopattern['Solve Time Sec'].tolist()]    # no pattern sudokus

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)

ax1.set_title('No Pattern')
ax1.set_ylabel('Solve Time in Seconds')
parts1 = ax1.violinplot(
        data1, showmeans=False, showmedians=False,
        showextrema=False)

# fill with colors
colors = ['lightgreen', 'lightblue']
for bplot in parts1:
    for patch, color in zip(parts1['bodies'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(1)
    
quartile1, medians, quartile3 = np.percentile(data1, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data1, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax1.scatter(inds, medians, marker='o', color='white', s=8, zorder=3)
ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax1.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

ax2.set_title('Pattern')
parts2 = ax2.violinplot(
        data2, showmeans=False, showmedians=False,
        showextrema=False)

# Fill with colors
colors = ['lightgreen', 'lightblue']
for bplot in parts2:
    for patch, color in zip(parts2['bodies'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(data2, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data2, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=8, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# Set style for the axes
labels = ['DPLL', 'CDCL']
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()
