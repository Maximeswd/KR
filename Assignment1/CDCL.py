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
