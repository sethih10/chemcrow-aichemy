"""
test_cof_multi_bo.py

Explainable test for the COFMultiObjectiveBO ChemCrow tool.

This script:
  - Builds the clean ChemCrow instance.
  - Constructs a JSON config for multi-objective BO over your COF dataset.
  - Asks the agent (via natural language) to call COFMultiObjectiveBO exactly once
    with that JSON and to summarise the results (Pareto front + BO suggestions).

Assumptions:
  - clean_chemcrow_minimal.py defines build_clean_chemcrow().
  - COFMultiObjectiveBO has been imported and registered as a tool
    in build_clean_chemcrow().
  - Data layout:

      CIF directory:
        C:\\Users\\brown\\Downloads\\COF_crystals\\crystals

      Descriptor CSV:
        C:\\Users\\brown\\Downloads\\COF_crystals\\cof_descriptors.csv

      Property CSV:
        C:\\Users\\brown\\Downloads\\COF_crystals\\gcmc_calculations.csv

  - We maximise two objectives:

      1. "⟨N⟩ (mmol/g)"      # uptake per gram
      2. "selectivity Xe/Kr" # separation performance
"""

import os

from clean_chemcrow_minimal import build_clean_chemcrow


# --- Local paths (Windows) --- #

CIF_DIR = r"C:\Users\brown\Downloads\COF_crystals\crystals"
DESCRIPTOR_CSV = r"C:\Users\brown\Downloads\COF_crystals\cof_descriptors.csv"
PROPERTY_CSV = r"C:\Users\brown\Downloads\COF_crystals\gcmc_calculations.csv"


def main():
    # Warn if OPENAI key is missing (ChemCrow will probably need it)
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Some tools may fail.\n")

    chemcrow = build_clean_chemcrow()

    # Escape backslashes for JSON
    cif_dir_json = CIF_DIR.replace("\\", "\\\\")
    desc_csv_json = DESCRIPTOR_CSV.replace("\\", "\\\\")
    prop_csv_json = PROPERTY_CSV.replace("\\", "\\\\")

    # JSON config for COFMultiObjectiveBO
    # Two objectives to MAXIMISE:
    #   - "⟨N⟩ (mmol/g)"      (uptake)
    #   - "selectivity Xe/Kr" (separation)
    cof_mobo_config_json = (
        "{"
        f"\"cif_dir\": \"{cif_dir_json}\", "
        f"\"descriptor_csv\": \"{desc_csv_json}\", "
        f"\"property_csvs\": [\"{prop_csv_json}\"], "
        "\"target_properties\": ["
        "\"⟨N⟩ (mmol/g)\", "
        "\"selectivity Xe/Kr\""
        "], "
        "\"id_column\": \"crystal_name\", "
        "\"property_agg\": \"mean\", "
        "\"top_k\": 15, "
        "\"n_weight_samples\": 6"
        "}"
    )

    # Natural-language test prompt that tells the agent exactly what to do
    test_prompt = f"""
You are running inside a ChemCrow environment with a custom tool:

  - COFMultiObjectiveBO

The COFMultiObjectiveBO tool performs **multi-objective Bayesian optimisation**
over a library of COFs. It expects a SINGLE STRING input which MUST be a JSON
blob with the following fields:

  - "cif_dir": directory containing CIF files (optional but recommended)
  - "descriptor_csv": CSV with numeric descriptors per crystal
  - "property_csvs": list of CSV files with properties per crystal
  - "target_properties": list of property column names to MAXIMISE
  - "id_column": common crystal ID column across all CSVs
  - "property_agg": aggregation method across repeated measurements ("mean", "max", "min")
  - "top_k": number of BO suggestions to return
  - optional: "weights" and/or "n_weight_samples" for scalarisation

For this test, the dataset is:

  CIF directory:
    {CIF_DIR}

  Descriptor CSV:
    {DESCRIPTOR_CSV}

  Property CSV:
    {PROPERTY_CSV}

We will MAXIMISE TWO objectives taken from the property CSV:

  1. "⟨N⟩ (mmol/g)"      – uptake per gram.
  2. "selectivity Xe/Kr" – Xe/Kr separation performance.

Here is the EXACT JSON config you must pass to COFMultiObjectiveBO
as its tool input (do NOT modify this string):

  {cof_mobo_config_json}

Your task:

1. Briefly explain, in your own words, what these two objectives represent
   physically and why they are sensible to maximise together in a COF screen
   (no tools needed for this part).

2. Call COFMultiObjectiveBO **exactly once**, using the JSON string above
   as the tool input.

3. After the tool returns, summarise the result:
   - How many COFs were considered in total.
   - Which COFs currently form the **observed Pareto front**, listing for each:
       * crystal identifier
       * values of both objectives ("⟨N⟩ (mmol/g)" and "selectivity Xe/Kr").
   - The **top BO suggestions** (up to 15), including:
       * crystal identifier
       * predicted values and uncertainties for each objective
       * EI score (or ranking) used to select them.

4. Discuss briefly:
   - How the BO suggestions differ from the current Pareto front.
   - Any interesting trade-offs (e.g. high uptake but moderate selectivity,
     or vice versa).
   - Why this scalarisation-based approach is a reasonable approximation
     to multi-objective BO, and what its limitations are.

Important:
- Do NOT fabricate crystal names or numeric values; rely ONLY on what
  COFMultiObjectiveBO returns.
- If the tool fails (e.g. missing files or wrong column names), clearly report
  the error and stop.
"""

    print("\n=== TEST PROMPT (COFMultiObjectiveBO) ===\n")
    print(test_prompt)
    print("\n=== MODEL RESPONSE ===\n")

    answer = chemcrow.run(test_prompt)
    print(answer)


if __name__ == "__main__":
    main()
