# Packages
import pandas as pd
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats

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


### Definition hard sudokus 1: Number of givens based
# - Group of hard sudokus defined on 17 givens per sudoku
# - Group of easy sudokus defined on 18, 19, 20, 21, 22 givens per sudoku
def1_hard = d_sorted[0][1]
def1_easy = d_sorted[1][1] and d_sorted[2][1] and d_sorted[3][1] and d_sorted[4][1] 



# CDCL (Conflict Driven Clause Learning) based SAT SOLVER , implemented by kkapil@iitg.ac.in
# Key Features : 
# 1. 2- Literal Watch : Two literal watch data structure based bcp
# 2. Decision heuristics : VSIDS (Variable State Independent Decaying Sum) 
# 3. Random Restarts (with decaying probability)
# In[ ]:



# Function to read clauses from cnf file "fname".
def read_clause (fname):
    f = open(fname,"r")
    read_count = 0
    clauses = []
    readflag = 1                 # =1, indicate new clause. The definition of a clause may extend beyond a single line of text as per standard format 
    for x in f :                 # for each line
        if x[0] != 'c' :         #if not comment
            if x[0] == 'p' :     #if 'p' i.e problem def. line
                x = x.split()
                num_var = int(x[2])      #read number of variables and clauses
                num_claus = int(x[3])
            else :                     #if clause
                if readflag == 1 :     #new clause starts
                    newclause = []
                x = x.split()
                for j in x :
                    i = int(j)
                    readflag = 0
                    if i == 0 :          #if 0,i.e clause end, push this clause to list clauses and start a new clause
                        clauses.append(newclause)
                        readflag = 1
                        read_count += 1
                        break
                    if -i in newclause:
                        readflag = 1
                        read_count += 1
                        break
                    if i not in newclause:
                        newclause.append(i)       #while Integer not zero append to current clause
                        
                    #Following conditions are for checking Inconsistent or Invalid input cnf file
                    if i > num_var :
                        print("Illegal variable number "+str(i))
                        return 0,num_var,num_claus
                    if read_count >= num_claus :
                        print("Number of clauses exceeds definition")
                        return 0,num_var,num_claus
    return 1,num_var,num_claus,clauses


# Following functions are for hueristic VSIDS

# Initiatilaztion  : Generated counter for number of times a literal appears
def VSIDS_init(clauses,num_var):
    counter = {}
    for x in range(-num_var,num_var+1):
        counter[x]=0
    for clause in clauses:
        for literal in clause:
            counter[literal] += 1
    return counter

# conflict : Incerements counter of literalts in conflict clause to increase there chances of getting selected
def VSIDS_conflict(counter,conflictClause):
    for literal in conflictClause:
        counter[literal]+=1
    return counter

# decay : Counter is reduced by 5% for all literals at each conflict
def VSIDS_decay(counter,num_var):
    for i in range(-num_var,num_var+1):
        counter[i]=counter[i]*95/100
    return counter

# decide : Picks a Variable NOT yet in M based on max counter value
def VSIDS_decide(counter,M,num_var):
    max=0
    var=0
    for x in range(-num_var,num_var+1):
        if counter[x]>=max and x not in M and -x not in M:
                max=counter[x]
                var=x
    return var


# BCP and Unit Propogation in ONLY used in beginning to get rid of unit clauses and Their implications
def bcp(clauses, literal):                    #Boolean Constant Propagation on Literal
    new_claus_set = [x[:] for x in clauses]   #Using SLicing Technique: Fastest available in Python
    for x in reversed(new_claus_set):
        if literal in x:                      #if clause satified ,
            new_claus_set.remove(x)                    #Remove that clause
        if -literal in x:                     #if -literal present , remaining should satisfy . Hence,
            x.remove(-literal)                         #Remove -literal from that clause
            if not x:                         #if this makes a clause Empty , UNSAT
                return -1
    return new_claus_set

def unit_propagation(clauses):               # Propogate Unit Clauses and add implications to M
    assignment = []
    flag=1
    while flag!=0:                           #till Implications are found
        flag=0
        for x in clauses:                    #for each clause
            if len(x) == 1 :                 # if UNIT clause , propagate and add to assignment
                unit=x[0]
                clauses = bcp(clauses, unit) 
                assignment += [unit]
                flag=1
            if clauses == -1:                #if UNSAT after propogate, return -1
                return -1, []
            if not clauses:                   
                return clauses, assignment
    return clauses, assignment


def CDCL_solve(clauses,num_var):
    decide_pos = []                             # for Maintaing Decision Level
    M = []                                      # Current Assignments and implications
    clauses,M = unit_propagation(clauses)                        # Initial Unit Propogation : if conflict - UNSAT
    if clauses == -1 :
        return -1,0,0,0,0                                        # UNSAT
    back=M[:]                                                    # Keep Initialization Backup for RESTART
    counter = VSIDS_init(clauses,num_var)                        # Initialize Heuristic counter
    
    # Initialize TWO LITERAL WATCH data Structure :
    literal_watch,clauses_literal_watched = create_watchList(clauses,M,num_var)

    probability=0.9                                             # Random Restart Probability ,  Decays with restarts
    Restart_count = Learned_count = Decide_count = Imp_count = 0
    
    while not all_vars_assigned(num_var , len(M)) :             # While variables remain to assign
        variable = VSIDS_decide(counter,M,num_var)                      # Decide : Pick a variable
        Decide_count += 1
        progressBar(len(M),num_var)                             # print progress    
        assign(variable,M,decide_pos)
        conflict,literal_watch = two_watch_propogate(clauses,literal_watch,clauses_literal_watched,M,variable)         # Deduce by Unit Propogation
        
        
        while conflict != -1 :
            VSIDS_conflict(counter,conflict)                    # Incerements counter of literalts in conflict
            counter=VSIDS_decay(counter,num_var)                # Decay counters by 5%

            Learned_c = Analyze_Conflict(M, conflict,decide_pos)      #Diagnose Conflict

            dec_level = add_learned_clause_to(clauses,literal_watch,clauses_literal_watched,Learned_c,M) #add Learned clause to all data structures
            Learned_count += 1
            jump_status,var,Imp_count = Backjump(M, dec_level, decide_pos,Imp_count)      #BackJump to decision level

            if jump_status == -1:                                     # UNSAT
                return -1,Restart_count,Learned_count,Decide_count,Imp_count
            M.append(var)                                             # Append negation of last literal after backjump
            
            probability,Restart_count = RandomRestart(M,back,decide_pos,probability,Restart_count)        #Random Restart
            conflict,literal_watch = two_watch_propogate(clauses,literal_watch,clauses_literal_watched,M,var)

            
    #Reaches here if all variables assigned. 

    return M,Restart_count,Learned_count,Decide_count,Imp_count
    

def create_watchList(clauses,M,num_var):          # Create the 2-literal watch data structure
    literal_watch = {}                    # Will contain the main Literal-> Clause number mapping
    clauses_literal_watched = []          # The reverse,i.e. Clause-> Literal mapping
    for i in range (-num_var,num_var+1):
        literal_watch[i] = []
    for i in range (0,len(clauses)):      #for each clause pick two literals
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
        clauses_literal_watched.append(newc)       #add both to watched of that clause 
        literal_watch[A].append(i)                 #add clause to watch of those literals
        literal_watch[B].append(i)
    return literal_watch,clauses_literal_watched



# Function to propogate using 2-literal watch

def two_watch_propogate(clauses,literal_watch,clauses_literal_watched,M,variable): 
    prop_list = [variable]             # add current change to list of updates
    while len(prop_list) != 0 :        # while updates remain to propogate
        variable = prop_list.pop()     #pick one variable
        for affected_claus_num in reversed(literal_watch[-variable]) : #for all clauses in its watch list
            affected_claus = clauses[affected_claus_num][:]
            A = clauses_literal_watched[affected_claus_num][0]
            B = clauses_literal_watched[affected_claus_num][1]
            A_prev=A
            B_prev=B
            status,M,A,B,unit = check_status(affected_claus,M,A,B)     # check status of each clause
            if status == "Unit" :
                prop_list.append(unit)
                M.append(unit)                                         # if unit, add to updates
            elif status == "Unsatisfied":                              #if unsat, return conflict clause
                return affected_claus,literal_watch
                                                                       #else the clause is still unresolve, remove from current add to proper watch
            literal_watch [A_prev].remove(affected_claus_num)
            literal_watch [B_prev].remove(affected_claus_num)
            clauses_literal_watched[affected_claus_num][0] = A
            clauses_literal_watched[affected_claus_num][1] = B
            literal_watch [A].append(affected_claus_num)
            literal_watch [B].append(affected_claus_num)
            
    return -1,literal_watch


def check_status(clause,M,A,B):
    unit = 0
    if A in M or B in M:                   # if one watch satisfied, nothing to do 
        return "Satisied",M,A,B,unit
    sym=[]                                  # symbols not defined yet
    for literal in clause:                  # find symbols not defined
        if -literal not in M:
            sym.append(literal)
        if literal in M :
            if -A not in M :
                return "Satisied",M,A,literal,unit
            return "Satisied",M,literal,B,unit
    if len(sym) == 1:                              # if one such symbol -> Unit Clause
        return "Unit",M,A,B,sym[0]
    if len(sym) == 0:                              # if no such symbol -> Unsatisfied (conflict) clause
        return "Unsatisfied",M,A,B,unit
    else :
        return "Unresolved",M,sym[0],sym[1],unit   # else return two new unsatisfied variables to use for Literal_watch


def RandomRestart(M,back,decide_pos,probability,Restart_count):  
    if random.random() < probability :          # If Generated random probability less than current : RESTART
        M = back[:]
        decide_pos = []
        probability *= 0.5                      # Decay next Restart probability by 50%
        Restart_count += 1
        if probability < 0.001 :
            probability = 0.2
        if Restart_count > len(M) + 10:         #avoid restarts if already restarted many times
            probability=0
    return probability,Restart_count


def verify(M,clauses) :                  # Verify the Solution in M for SAT
    for c in clauses :                   # for each clause
        flag = 0
        for lit in c:
            if lit in M:                 # atleast one literal should be true
                flag = 1
                break
        if flag == 0:
            return False
    return True


def Analyze_Conflict(M, conflict,decide_pos):  #for simplicity : ALL DECISIONs made till now are a Learned Clause 
    learn = []
    for x in decide_pos:
        learn.append(-M[x])
    return learn


def all_vars_assigned(num_var ,M_len):        # Returns True if all variables already assigne , False otherwise
    if M_len >= num_var:
        return True
    return False


def assign(variable,M,decide_pos):             # Adds the decision literal to M and correponding update to decision level
    decide_pos.append(len(M))
    M.append(variable)


def add_learned_clause_to(clauses,literal_watch,clauses_literal_watched,Learned_c,M):
    if len(Learned_c) == 0:
        return -1
    if len(Learned_c) == 1:             # if unit clause is learnt : add it as a decision 
        M.append(Learned_c[0])
        return 1,Learned_c[0]
    clauses.append(Learned_c)           # for others, add two literals A,B to literal watch data structure
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


def Backjump(M, dec_level, decide_pos,Imp_count):         #BackJump to decision level by deleting decisions from M and decision positions
    Imp_count = Imp_count + len(M) - len(decide_pos)
    if not decide_pos:
        return -1,-1,Imp_count
    dec_level = decide_pos.pop()
    literal = M[dec_level]
    del M[dec_level:]
    return 0,-literal,Imp_count



def progressBar(current, total, barLength = 20) :        # Print progress bar. Just to givee feel of work being done
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress (num_var:may backtrack): [%s%s] %d ' % (arrow, spaces, current), end='\r')


# Function to solve per sudoku
def solve(number):
    
    # Read the input file
    a,num_var,num_claus,clauses = read_clause("/Users/gast/Desktop/VS code/Projects/Knowledge Representation/sudoku%s.cnf"%(number))                  # Read from input file
    
    # Print if the file could be read 
    if a == 1:                                                       # Status of reading input
        print("Successfully read the file")
    else:
        print("Could not read the file")
        return

    startSolve = time.process_time()
    solution = CDCL_solve(clauses, num_var)                            # Solve CNF by CDCL
    EndSolve = time.process_time()

    if solution[0] != -1:

        result_dict = {
            "Satisfied": "SAT", 
            "Restarts": solution[1],
            "Learned Clauses": solution[2],
            "Decisions": solution[3],
            "Implications": solution[4], 
            "Solve Time Sec": EndSolve-startSolve, 
            }
    else :
        result_dict = {
            "Satisfied": "UNSAT", 
            "Restarts": solution[1],
            "Learned Clauses": solution[2],
            "Decisions": solution[3],
            "Implications": solution[4], 
            "Solve Time Sec": EndSolve-startSolve, 
            }
    
    return result_dict


def get_results(lis, n):
    results = []
    inds = random.sample(lis, n)
    for i in inds:
        results.append(solve(i))
    return results

# Get the information of all the solved sudokus per group (hard vs easy)
res_hardG = get_results(d_sorted[0][1], len(d_sorted[0][1]))
res_easyG = get_results(d_sorted[1][1] and d_sorted[2][1] and d_sorted[3][1] and d_sorted[4][1], len(d_sorted[0][1]))

# Put them into a dataframe
df_hardG = pd.DataFrame(res_hardG)
df_easyG = pd.DataFrame(res_easyG)

# Conduct Welch's t-Test and print the result
np.var(df_hardG['Solve Time Sec']), np.var(df_easyG['Solve Time Sec'])
stats.ttest_ind(df_hardG['Solve Time Sec'], df_easyG['Solve Time Sec'], equal_var = False)
