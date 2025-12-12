import sys
sys.path.append('/scratch/work/sethih1/Crow/chemcrow-aichemy/')


from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from chemcrow.agents import ChemCrow
chem_model = ChemCrow(model="gpt-4-0613", temp=0.1, verbose=True) #, streaming=False)


import json
import io
from contextlib import redirect_stdout

# Capture the printed output
f = io.StringIO()
with redirect_stdout(f):
    response = chem_model.run("""
Step 1. Read the CSV file at '/scratch/work/sethih1/Crow/crystal_selectivity_clean.csv'.
Step 2. Extract the column names of the csv file. 
Step 3. Save the column names for the next step.
Step 4. Use the BayesianOptimizeData tool with extracted actual column names and use it as input to the BayesianOptimizer tool. Inputs are target (selectivity), features(like void_fraction, etc.) and label names (crystal names). Use the tool to train a model using the extracted features as input and 'selectivity_Xe/Kr' as the target.
Step 5. Output the predicted 'crystal_name' only.""")

printed_output = f.getvalue()
f.close()

# Prepare a JSON object
log_json = {
    "response_returned": response,       # if response is structured
    "printed_output": printed_output     # everything that was printed
}

# Save as JSON
with open("/scratch/work/sethih1/Crow/chemcrow-aichemy/chemcrow_log.json", "w") as outfile:
    json.dump(log_json, outfile, indent=4)


chem_model.run("""Step 1. Read the CSV file at '/scratch/work/sethih1/Crow/crystal_selectivity_clean.csv'.
Step 2. Extract the column names of the CSV file. 
Step 3. Save the column names for the next step.
Step 4. Use the BayesianOptimizeData tool with extracted actual column names and use it as input to the BayesianOptimizer tool. Inputs are target (selectivity), features (like void_fraction, etc.) and label names (crystal names). Use the tool to train a model using the extracted features as input and 'selectivity_Xe/Kr' as the target.
Step 5. Output the predicted 'crystal_name' only, and make sure to remove any '.cif' extension from the names.
Step 6. For each predicted 'crystal_name', use MPStructureQuery  tool to fetch the crystal structure 
Step 7. Use CoordinationAnalysis tool to generate coordination numbers and nearest neighbor information for each site

""")
