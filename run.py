import sys
sys.path.append('/scratch/work/sethih1/Crow/chemcrow-aichemy/')


from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from chemcrow.agents import ChemCrow
chem_model = ChemCrow(model="gpt-4-0613", temp=0.1, verbose=True) #, streaming=False)


import sys

log_file = "/scratch/work/sethih1/Crow/chemcrow_log.txt"
sys.stdout = open(log_file, "w")  # Redirect print output to file

chem_model.run("""
Step 1. Read the CSV file at '/scratch/work/sethih1/Crow/crystal_selectivity_clean.csv'.
Step 2. Extract the 'selectivity_Xe/Kr' column and all the remaining columns except 'crystal_name'.
Step 3. Save the names of the columns for reference.
Step 4. Use the BayesianOptimizeData tool with extracted actual column names and use it as input to the BayesianOptimizer tool. With input as target, features and label names to train a model using the extracted features as input and 'selectivity_Xe/Kr' as the target.
Step 5. Output the predicted 'crystal_name' values along with their corresponding 'selectivity_Xe/Kr'.

""")

sys.stdout.close()  # Restore stdout later if needed
