import time
import random
import pandas as pd
import numpy as np
import scipy.stats as stats


def decide(M, num_var):
    var = 0
    for x in range(-num_var, num_var + 1):
        if x not in M and -x not in M:
            var = x
    return var


def check_status(clause, M):
    unit = 0
    sym = []  # symbols not defined yet
    for literal in clause:  # find symbols not defined
        if -literal not in M:
            sym.append(literal)
        if literal in M:
            return "Satisfied", M, unit
    if len(sym) == 1:  # if one such symbol -> Unit Clause
        return "Unit", M, sym[0]
    if len(sym) == 0:  # if no such symbol -> Unsatisfied (conflict) clause
        return "Unsatisfied", M, unit
    else:
        return "Unresolved", M, unit  # else return two new unsatisfied variables to use for Literal_watch


def propagate(clauses, M, variable):
    prop_list = [variable]  # add current change to list of updates
    while len(prop_list) != 0:  # while updates remain to propogate
        variable = prop_list.pop()  # pick one variable
        for affected_claus_num in reversed(clauses[-variable]):  # for all clauses in its watch list
            affected_claus = clauses[affected_claus_num][:]
            status, M, unit = check_status(affected_claus, M)  # check status of each clause
            if status == "Unit":
                prop_list.append(unit)
                M.append(unit)  # if unit, add to updates
            elif status == "Unsatisfied":  # if unsat, return conflict clause
                return affected_claus
    return -1


def Backjump(M, decide_pos,
             Imp_count):  # BackJump to decision level by deleting decisions from M and decision positions
    Imp_count = Imp_count + len(M) - len(decide_pos)
    if not decide_pos:
        return -1, -1, Imp_count
    dec_level = decide_pos.pop()
    literal = M[dec_level]
    del M[dec_level:]
    return 0, -literal, Imp_count


def learn_clauses(M, decide_pos):  # for simplicity : ALL DECISIONs made till now are a Learned Clause
    learn = []
    for x in decide_pos:
        learn.append(-M[x])
    return learn


def dpll(clauses, num_var):
    decide_pos = []  # for Maintaing Decision Level
    clauses, M = unit_propagation(clauses)  # Initial Unit Propogation : if conflict - UNSAT
    if clauses == -1:
        return -1, 0, 0, 0, 0  # UNSAT

    Learned_count = Decide_count = Imp_count = 0

    while not all_vars_assigned(num_var, len(M)):  # While variables remain to assign
        variable = decide(M, num_var)  # Decide : Pick a variable
        Decide_count += 1
        progressBar(len(M), num_var)  # print progress
        assign(variable, M, decide_pos)

        conflict = propagate(clauses, M, variable)  # Deduce by Unit Propogation

        while conflict != -1:
            Learned_c = learn_clauses(M, decide_pos)

            add_learned_clause_to(clauses, Learned_c, M)  # add Learned clause to all data structures
            Learned_count += 1
            jump_status, var, Imp_count = Backjump(M, decide_pos, Imp_count)  # BackJump to decision level

            if jump_status == -1:  # UNSAT
                return -1, Learned_count, Decide_count, Imp_count
            M.append(var)  # Append negation of last literal after backjump
            conflict = propagate(clauses, M, variable)  # Deduce by Unit Propogation

    # Reaches here if all variables assigned.
    return M, Learned_count, Decide_count, Imp_count


def solve(number):
    # Read the input file
    a, num_var, num_claus, clauses = read_clause(
        "./sudoku%s.cnf" % number)  # Read from input file

    # Print if the file could be read
    if a == 1:  # Status of reading input
        print("Successfully read the file")
    else:
        print("Could not read the file")
        return

    startSolve = time.process_time()
    solution = dpll(clauses, num_var)  # Solve CNF by DPLL
    EndSolve = time.process_time()

    if solution[0] != -1:

        result_dict = {
            "Satisfied": "SAT",
            # "Restarts": solution[1],
            "Learned Clauses": solution[1],
            "Decisions": solution[2],
            "Implications": solution[3],
            "Solve Time Sec": EndSolve - startSolve,
        }
    else:
        result_dict = {
            "Satisfied": "UNSAT",
            # "Restarts": solution[1],
            "Learned Clauses": solution[1],
            "Decisions": solution[2],
            "Implications": solution[3],
            "Solve Time Sec": EndSolve - startSolve,
        }

    return result_dict


# Function to read clauses from cnf file "fname".
def read_clause(fname):
    f = open(fname, "r")
    read_count = 0
    clauses = []
    readflag = 1  # =1, indicate new clause. The definition of a clause may extend beyond a single line of text as per standard format
    for x in f:  # for each line
        if x[0] != 'c':  # if not comment
            if x[0] == 'p':  # if 'p' i.e problem def. line
                x = x.split()
                num_var = int(x[2])  # read number of variables and clauses
                num_claus = int(x[3])
            else:  # if clause
                if readflag == 1:  # new clause starts
                    newclause = []
                x = x.split()
                for j in x:
                    i = int(j)
                    readflag = 0
                    if i == 0:  # if 0,i.e clause end, push this clause to list clauses and start a new clause
                        clauses.append(newclause)
                        readflag = 1
                        read_count += 1
                        break
                    if -i in newclause:
                        readflag = 1
                        read_count += 1
                        break
                    if i not in newclause:
                        newclause.append(i)  # while Integer not zero append to current clause

                    # Following conditions are for checking Inconsistent or Invalid input cnf file
                    if i > num_var:
                        print("Illegal variable number " + str(i))
                        return 0, num_var, num_claus
                    if read_count >= num_claus:
                        print("Number of clauses exceeds definition")
                        return 0, num_var, num_claus
    return 1, num_var, num_claus, clauses


def bcp(clauses, literal):  # Boolean Constant Propagation on Literal
    for x in clauses:
        if literal in x:  # if clause satified ,
            clauses.remove(x)  # Remove that clause
        if -literal in x:  # if -literal present , remaining should satisfy . Hence,
            x.remove(-literal)  # Remove -literal from that clause
            if not x:  # if this makes a clause Empty , UNSAT
                return -1
    return clauses


def unit_propagation(clauses):  # Propogate Unit Clauses and add implications to M
    assignment = []
    flag = 1
    while flag != 0:  # till Implications are found
        flag = 0
        for x in clauses:  # for each clause
            if len(x) == 1:  # if UNIT clause , propagate and add to assignment
                unit = x[0]
                clauses = bcp(clauses, unit)
                assignment += [unit]
                flag = 1
            if clauses == -1:  # if UNSAT after propogate, return -1
                return -1, []
            if not clauses:
                return clauses, assignment
    return clauses, assignment


def all_vars_assigned(num_var, M_len):  # Returns True if all variables already assigned , False otherwise
    if M_len >= num_var:
        return True
    return False


def assign(variable, M, decide_pos):  # Adds the decision literal to M and correponding update to decision level
    decide_pos.append(len(M))
    M.append(variable)


def progressBar(current, total, barLength=20):  # Print progress bar. Just to givee feel of work being done
    percent = float(current) * 100 / total
    arrow = '-' * int(percent / 100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))
    print('Progress (num_var:may backtrack): [%s%s] %d ' % (arrow, spaces, current), end='\r')


def get_results(lis, n):
    results = []
    inds = random.sample(lis, n)
    for i in inds:
        results.append(solve(i))
    return results


def add_learned_clause_to(clauses, Learned_c, M):
    if len(Learned_c) == 0:
        return -1
    if len(Learned_c) == 1:  # if unit clause is learnt : add it as a decision
        M.append(Learned_c[0])
        return 1, Learned_c[0]
    clauses.append(Learned_c)  # for others, add two literals A,B to literal watch data structure
    return 0


with_pattern = [2130, 1374, 2185, 2384, 2179, 2139, 2091, 2389, 2377, 529, 2190, 1449, 1373, 1942, 1300, 2131, 1828,
                320, 1117, 2382, 1410, 1687, 559, 1851, 2132, 573, 2372, 2129, 2366, 125, 2369, 2394, 2080, 2146, 1625,
                568, 1964, 2371, 1401, 458, 2193, 2376, 631, 1038, 2124, 2390, 318, 2370, 2380, 2385, 1009, 1989, 2144,
                465, 875, 2103, 650, 1094, 2088, 2143, 497, 2052, 2374, 321, 2162, 846, 2367, 2375, 2392, 1368, 1098,
                2076, 1626, 2053, 2069, 1605, 2393, 1472, 2373, 328, 1686, 2127, 1644, 1059, 1372, 1347, 2395, 1994,
                2140, 2386, 1478, 1486, 1855, 396, 1688, 2104]

without_pattern = [59, 78, 92, 100, 101, 102, 113, 115, 143, 144, 145, 146, 291, 294, 310, 329, 330, 401, 421, 467, 521,
                   551, 552, 553, 554, 558, 604, 651, 1047, 1062, 1080, 1112, 1115, 1118, 1138, 1335, 1350, 1547, 1556,
                   1567, 1611, 1616, 1618, 1643, 1646, 1919, 1959, 1962, 1969, 1970, 1972, 1973, 1997, 2034, 2035, 2036,
                   2038, 2039, 2040, 2066, 2081, 2106, 2113, 2114, 2115, 2157, 2168, 2244, 2365, 2368, 2378, 2379, 2381,
                   2383, 2387, 2388, 2391]

if __name__ == '__main__':

    res_with_pattern = get_results(with_pattern, len(with_pattern))
    print(res_with_pattern)
    res_without_pattern = get_results(without_pattern, len(without_pattern))
    print(res_without_pattern)

    output_file_with_pattern = open('results_with_pattern.txt', 'w')
    output_file_with_pattern.write(str(res_with_pattern))

    output_file_without_pattern = open('results_without_pattern.txt', 'w')
    output_file_without_pattern.write(str(res_without_pattern))

    # Put them into a dataframe
    df_with_pattern = pd.DataFrame(res_with_pattern)
    df_without_pattern = pd.DataFrame(res_without_pattern)

    # Conduct Welch's t-Test and print the result
    np.var(df_with_pattern['Solve Time Sec']), np.var(df_without_pattern['Solve Time Sec'])
    print(stats.ttest_ind(df_with_pattern['Solve Time Sec'], df_without_pattern['Solve Time Sec'], equal_var=False))
