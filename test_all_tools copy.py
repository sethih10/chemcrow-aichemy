# test_all_tools.py
"""
End-to-end test script that tells ChemCrow to exercise all custom tools:

  - ListCrystalCIFs
  - VastraVisualise
  - MotifDecomposition
  - MotifComparison
  - ArxivLiteratureSearch

Assumes:
  - clean_chemcrow_minimal.py defines build_clean_chemcrow, DEFAULT_CRYSTAL_DIR, TARGET_CIF.
  - motifs_cof_simple.json exists at DEFAULT_MOTIF_LIB.
"""

import os

from clean_chemcrow_minimal import (
    build_clean_chemcrow,
    DEFAULT_CRYSTAL_DIR,
    TARGET_CIF,
    DEFAULT_MOTIF_LIB,
)


def main():
    # Make sure OpenAI key is visible
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Tools that call OpenAI may fail.\n")

    chem_model = build_clean_chemcrow()

    target_cif_str = str(TARGET_CIF)
    crystal_dir_str = str(DEFAULT_CRYSTAL_DIR)
    motif_lib_str = str(DEFAULT_MOTIF_LIB)

    # Windows JSON escaping for paths
    target_cif_json = target_cif_str.replace("\\", "\\\\")
    motif_lib_json = motif_lib_str.replace("\\", "\\\\")

    # JSON payload examples we want the agent to send
    motif_decomp_json = (
        "{"
        f"\"mode\": \"all\", "
        f"\"cif_path\": \"{target_cif_json}\", "
        f"\"motif_library_path\": \"{motif_lib_json}\", "
        "\"allow_overlap\": true"
        "}"
    )

    motif_compare_json_template = (
        "{{"
        "\"cif_path_1\": \"{cif1}\", "
        "\"cif_path_2\": \"{cif2}\", "
        f"\"motif_library_path\": \"{motif_lib_json}\", "
        "\"allow_overlap\": true"
        "}}"
    )

    test_prompt = f"""
You are wired into a ChemCrow environment with the following tools available
(at least):

  - Python_REPL
  - Wikipedia
  - Mol2CAS
  - PatentCheck
  - SMILES2Weight
  - FunctionalGroups

  - ListCrystalCIFs
  - VastraVisualise
  - MotifDecomposition
  - MotifComparison
  - ArxivLiteratureSearch

Your job is to exercise ALL of the custom tools at least once in a sensible way.

The CIF directory is:

  {crystal_dir_str}

A specific CIF of interest is:

  {target_cif_str}

The motif library (simple COF motifs) is at:

  {motif_lib_str}

Follow these steps, using tool calls explicitly:

1) Call **ListCrystalCIFs** once (with any string, e.g. empty) to list the
   available CIF files. Report the list back to me.

2) Pick a CIF file from the list. IF the file
   "{target_cif_str}"
   is present, prefer that one. Use that full absolute path in all later steps.

3) Call **MotifDecomposition** once on that CIF with EXACTLY the following JSON
   string as the input (do not alter it other than inserting the correct slashes):

   {motif_decomp_json}

   Then:
   - Summarise how many motif instances were found, grouped by motif_name.
   - State how many unassigned_sites there are.

4) Call **VastraVisualise** once on the SAME CIF, with the full absolute path
   as the input (e.g. "{target_cif_str}").
   Wait for the tool result. Tell me where the PNG was written.

5) If there is at least one OTHER CIF in the directory, call **MotifComparison**
   once, comparing the primary CIF from step (3) with a second CIF.
   Use a JSON input of the form:

   {{"cif_path_1": "FULL_PATH_TO_FIRST_CIF",
     "cif_path_2": "FULL_PATH_TO_SECOND_CIF",
     "motif_library_path": "{motif_lib_str}",
     "allow_overlap": true}}

   Then:
   - List which motifs are shared between the two structures, with their counts.
   - List which motifs are unique to each structure.

6) Call **ArxivLiteratureSearch** once to answer the following question:

   "How are covalent organic frameworks (COFs) used for gas storage applications?"

   Answer based ONLY on ArxivLiteratureSearch outputs. Summarise in 3–5 sentences.

7) At the end, briefly confirm which tools you actually invoked
   (ListCrystalCIFs, MotifDecomposition, MotifComparison, VastraVisualise,
   ArxivLiteratureSearch) and what each one returned at a high level.

Important:
- Do NOT fabricate tool outputs; rely entirely on actual tool calls.
- If any tool fails (e.g. missing file, VESTA not installed, motif library missing),
  clearly report the error message and continue with the remaining steps.
"""

    print("\n=== TEST PROMPT (exercise all custom tools) ===\n")
    print(test_prompt)
    print("\n=== MODEL RESPONSE ===\n")

    answer = chem_model.run(test_prompt)
    print(answer)


if __name__ == "__main__":
    main()
