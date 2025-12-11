from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from chemcrow.agents import ChemCrow


# Create the ChemCrow agent
chem_model = ChemCrow(
    model="gpt-4.1-mini",  # any GPT-4 / GPT-4o-style name you have access to
    temp=0.1,
    streaming=False
)

# Ask it a question
answer = chem_model.run("What is the molecular weight of paracetamol? dont use Name2SMILES")
print(answer)
