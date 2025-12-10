"""
Test script for porosity calculation tools.

This demonstrates how to use the new Porosity and PoreSizeDistribution tools.
Note: Requires Zeo++ to be installed on the system.
"""

import os
import tempfile
from chemcrow.tools.porosity import PorosityCalculator, PoreSizeDistribution
from chemcrow.utils import validate_cif_file


def create_sample_cif():
    """Create a sample CIF file for testing (MOF-5 structure)."""
    cif_content = """data_MOF-5
_cell_length_a 20.711
_cell_length_b 20.711
_cell_length_c 20.711
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 227
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_occupancy
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_thermal_displace_type
_atom_site_B_iso_or_equiv
_atom_site_type_symbol
Zn1      1.0 0.0000 0.0000 0.0000 Biso 1.000 Zn
O1       1.0 0.0958 0.0958 0.0000 Biso 1.000 O
C1       1.0 0.1219 0.1219 0.0000 Biso 1.000 C
C2       1.0 0.1980 0.1980 0.0000 Biso 1.000 C
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
        f.write(cif_content)
        return f.name


def test_porosity_calculator():
    """Test the PorosityCalculator tool."""
    print("=" * 70)
    print("Testing Porosity Calculator Tool")
    print("=" * 70)
    
    # Create a sample CIF file
    cif_file = create_sample_cif()
    print(f"\nCreated sample CIF file: {cif_file}")
    
    # Validate the CIF file
    is_valid, msg = validate_cif_file(cif_file)
    print(f"CIF Validation: {msg}")
    
    if is_valid:
        # Try to calculate porosity
        calculator = PorosityCalculator()
        
        if calculator.zeopp_path:
            print(f"Zeo++ found at: {calculator.zeopp_path}")
            result = calculator._run(cif_file)
            print("\nPorosity Calculation Results:")
            print(result)
        else:
            print("\nZeo++ is not installed on this system.")
            print("To install Zeo++, run:")
            print("  conda install -c conda-forge zeo++")
            print("Or download from: http://zeo.chem.cmu.edu/")
            print("\nWithout Zeo++, this tool will return an error.")
            
            # Still test the error handling
            result = calculator._run(cif_file)
            print(f"\nExpected error: {result}")
    
    # Clean up
    os.unlink(cif_file)
    print("\n" + "=" * 70)


def test_pore_size_distribution():
    """Test the PoreSizeDistribution tool."""
    print("=" * 70)
    print("Testing Pore Size Distribution Tool")
    print("=" * 70)
    
    # Create a sample CIF file
    cif_file = create_sample_cif()
    print(f"\nCreated sample CIF file: {cif_file}")
    
    # Try to calculate pore size
    calculator = PoreSizeDistribution()
    
    if calculator.zeopp_path:
        print(f"Zeo++ found at: {calculator.zeopp_path}")
        result = calculator._run(cif_file)
        print("\nPore Size Distribution Results:")
        print(result)
    else:
        print("\nZeo++ is not installed on this system.")
        print("This tool requires Zeo++ to calculate pore sizes.")
    
    # Clean up
    os.unlink(cif_file)
    print("\n" + "=" * 70)


def test_invalid_file():
    """Test error handling with invalid files."""
    print("=" * 70)
    print("Testing Error Handling")
    print("=" * 70)
    
    calculator = PorosityCalculator()
    
    # Test with non-existent file
    result = calculator._run("/nonexistent/file.cif")
    print(f"\nNon-existent file error:\n{result}")
    
    # Test with invalid CIF
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
        f.write("This is not a valid CIF file")
        invalid_cif = f.name
    
    is_valid, msg = validate_cif_file(invalid_cif)
    print(f"\nInvalid CIF validation: {msg}")
    
    os.unlink(invalid_cif)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Porosity Calculation Tools Test Suite".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    test_invalid_file()
    test_porosity_calculator()
    test_pore_size_distribution()
    
    print("All tests completed!")
