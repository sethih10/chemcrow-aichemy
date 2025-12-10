#!/usr/bin/env python3
"""
Example usage of the new Porosity tools in ChemCrow.

This script demonstrates:
1. Direct tool usage
2. Agent-based usage (with LLM)
3. Advanced scenarios
"""

import os
from pathlib import Path


def example_direct_tool_usage():
    """Example 1: Direct tool usage without LLM."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Direct Tool Usage")
    print("="*70 + "\n")
    
    from chemcrow.tools.porosity import PorosityCalculator, PoreSizeDistribution
    from chemcrow.utils import validate_cif_file
    
    # Path to your CIF file
    cif_file = "path/to/your_mof.cif"
    
    # Step 1: Validate the CIF file
    print("Step 1: Validating CIF file...")
    is_valid, msg = validate_cif_file(cif_file)
    print(f"  Status: {msg}")
    
    if not is_valid:
        print("  ✗ CIF file is invalid. Check the file format.")
        return
    
    # Step 2: Calculate porosity
    print("\nStep 2: Calculating porosity...")
    porosity_calc = PorosityCalculator()
    porosity_result = porosity_calc._run(cif_file)
    print(porosity_result)
    
    # Step 3: Calculate pore size distribution
    print("\nStep 3: Analyzing pore size distribution...")
    pore_calc = PoreSizeDistribution()
    pore_result = pore_calc._run(cif_file)
    print(pore_result)


def example_agent_usage():
    """Example 2: Using tools with ChemCrow agent."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Using Tools with ChemCrow Agent")
    print("="*70 + "\n")
    
    # This requires OpenAI API key
    # Set it: export OPENAI_API_KEY="sk-..."
    
    # from chemcrow.agents import ChemCrow
    #
    # # Initialize agent
    # agent = ChemCrow(model="gpt-4-0613", temp=0.1)
    #
    # # Query 1: Simple porosity calculation
    # print("Query: Calculate porosity of MOF-5.cif")
    # result1 = agent.run("What is the porosity of MOF-5.cif?")
    # print(result1)
    #
    # # Query 2: Compare multiple structures
    # print("\nQuery: Compare porosities of multiple MOFs")
    # result2 = agent.run(
    #     "Compare the porosity and pore sizes of MOF-5.cif and HKUST-1.cif"
    # )
    # print(result2)
    #
    # # Query 3: Analysis request
    # print("\nQuery: Analyze suitability for CO2 capture")
    # result3 = agent.run(
    #     "Based on porosity analysis of MOF-5.cif, is this material suitable for CO2 capture?"
    # )
    # print(result3)
    
    print("(Agent examples require OPENAI_API_KEY to be set)")
    print("See code comments for full examples")


def example_batch_processing():
    """Example 3: Batch processing multiple MOF structures."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing Multiple MOFs")
    print("="*70 + "\n")
    
    from chemcrow.tools.porosity import PorosityCalculator
    from chemcrow.utils import validate_cif_file
    
    # List of CIF files to analyze
    mof_structures = [
        "MOF-5.cif",
        "HKUST-1.cif",
        "UiO-66.cif",
        "ZIF-8.cif",
    ]
    
    calculator = PorosityCalculator()
    results = {}
    
    print("Processing MOF structures...")
    print()
    
    for mof_file in mof_structures:
        if not os.path.exists(mof_file):
            print(f"⚠️  {mof_file}: File not found (skipping)")
            continue
        
        print(f"📊 Analyzing {mof_file}...")
        
        # Validate first
        is_valid, msg = validate_cif_file(mof_file)
        if not is_valid:
            print(f"   ✗ Invalid: {msg}")
            continue
        
        # Calculate
        result = calculator._run(mof_file)
        results[mof_file] = result
        print(f"   ✓ Analysis complete")
    
    # Summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    for mof, result in results.items():
        print(f"\n{mof}:")
        print("-" * 70)
        print(result)


def example_advanced_analysis():
    """Example 4: Advanced analysis workflow."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Advanced Analysis Workflow")
    print("="*70 + "\n")
    
    from chemcrow.tools.porosity import PorosityCalculator, PoreSizeDistribution
    from chemcrow.utils import validate_cif_file
    import re
    
    cif_file = "path/to/mof.cif"
    
    # Validate
    is_valid, msg = validate_cif_file(cif_file)
    if not is_valid:
        print(f"Invalid file: {msg}")
        return
    
    # Calculate both metrics
    porosity_calc = PorosityCalculator()
    pore_calc = PoreSizeDistribution()
    
    porosity_result = porosity_calc._run(cif_file)
    pore_result = pore_calc._run(cif_file)
    
    # Parse results (in production, use proper parsing)
    print("Analysis Results:")
    print("-" * 70)
    print(porosity_result)
    print()
    print(pore_result)
    
    # Application assessment
    print("\n" + "="*70)
    print("SUITABILITY ASSESSMENT")
    print("="*70)
    
    # Extract porosity fraction (example parsing)
    if "Porosity Fraction:" in porosity_result:
        # Would parse actual value in production
        print("\n✓ Gas Storage Applications:")
        print("  - CO2 capture: Good if porosity > 0.4")
        print("  - H2 storage: Good if large accessible volume")
        print("  - CH4 storage: Depends on pore size distribution")
    
    print("\n✓ Catalysis Applications:")
    print("  - Need pore diameters matched to reactants")
    print("  - Surface area important for active sites")
    
    print("\n✓ Separation Applications:")
    print("  - Pore size important for molecular sieving")
    print("  - Need to match kinetic diameters of molecules")


def example_error_handling():
    """Example 5: Proper error handling."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Error Handling Best Practices")
    print("="*70 + "\n")
    
    from chemcrow.tools.porosity import PorosityCalculator
    from chemcrow.utils import validate_cif_file
    
    calculator = PorosityCalculator()
    test_files = [
        "nonexistent.cif",
        "wrong_format.txt",
        "invalid.cif",
    ]
    
    for test_file in test_files:
        print(f"\nTesting: {test_file}")
        print("-" * 70)
        
        # Option 1: Validate first
        is_valid, msg = validate_cif_file(test_file)
        if not is_valid:
            print(f"✓ Validation caught issue: {msg}")
            continue
        
        # Option 2: Let tool handle it
        result = calculator._run(test_file)
        if "Error" in result:
            print(f"✓ Tool returned error: {result[:80]}...")
        else:
            print(f"✓ Success: {result[:80]}...")


def show_cif_format():
    """Example 6: CIF format reference."""
    print("\n" + "="*70)
    print("EXAMPLE 6: CIF File Format Reference")
    print("="*70 + "\n")
    
    example_cif = """data_MOF-5
_cell_length_a    20.711
_cell_length_b    20.711
_cell_length_c    20.711
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_cell_volume      8851.91
_symmetry_Int_Tables_number 227

loop_
_atom_site_label
_atom_site_occupancy
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_thermal_displace_type
_atom_site_B_iso_or_equiv
_atom_site_type_symbol

Zn1   1.0  0.0000  0.0000  0.0000  Biso 1.000 Zn
O1    1.0  0.0958  0.0958  0.0000  Biso 1.000 O
C1    1.0  0.1219  0.1219  0.0000  Biso 1.000 C
C2    1.0  0.1980  0.1980  0.0000  Biso 1.000 C"""
    
    print("Minimal CIF file structure:")
    print(example_cif)
    print("\nRequired fields:")
    print("  - _cell_length_a, _cell_length_b, _cell_length_c (Å)")
    print("  - _cell_angle_alpha, _cell_angle_beta, _cell_angle_gamma (degrees)")
    print("  - Atomic coordinates (loop section)")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Porosity Tools - Usage Examples".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Uncomment examples to run:
    
    # show_cif_format()
    # example_direct_tool_usage()
    # example_agent_usage()
    # example_batch_processing()
    # example_advanced_analysis()
    # example_error_handling()
    
    print("\n" + "="*70)
    print("To run examples, uncomment them in the __main__ section")
    print("="*70 + "\n")
    
    print("Available examples:")
    print("1. show_cif_format() - CIF file format reference")
    print("2. example_direct_tool_usage() - Using tools directly")
    print("3. example_agent_usage() - Using with ChemCrow agent")
    print("4. example_batch_processing() - Process multiple files")
    print("5. example_advanced_analysis() - Advanced workflows")
    print("6. example_error_handling() - Error handling patterns")
    print("\nEdit this file and uncomment examples to run them.")
