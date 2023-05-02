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
