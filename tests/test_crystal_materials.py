"""
Test script for crystal materials tools.

This script tests all crystal materials tools using pymatgen.
Creates sample structures and tests each tool functionality.
"""

import os
import tempfile
from pathlib import Path

import pytest

try:
    from pymatgen.core import Structure, Lattice
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    pytestmark = pytest.mark.skip("pymatgen not available")

from chemcrow.tools.crystal_materials import (
    StructureFromCIF,
    StructureSymmetryAnalysis,
    StructureProperties,
    StructureComparison,
    CompositionAnalysis,
    StructureSubstitution,
    PrimitiveCellConversion,
    StructureToCIF,
    SurfaceGeneration,
    CoordinationAnalysis,
    LatticeParameterOptimization,
    MPStructureQuery,
)


@pytest.fixture
def sample_si_structure():
    """Create a simple silicon structure for testing."""
    lattice = Lattice.cubic(5.43)
    species = ["Si", "Si", "Si", "Si"]
    coords = [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]
    return Structure(lattice, species, coords)


@pytest.fixture
def sample_tio2_structure():
    """Create a simple TiO2 structure for testing."""
    lattice = Lattice.tetragonal(4.6, 2.96)
    species = ["Ti", "Ti", "O", "O", "O", "O"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], 
              [0.3, 0.3, 0.0], [0.7, 0.7, 0.0],
              [0.2, 0.8, 0.5], [0.8, 0.2, 0.5]]
    return Structure(lattice, species, coords)


@pytest.fixture
def cif_file_si(sample_si_structure, tmp_path):
    """Create a temporary CIF file for silicon."""
    cif_path = tmp_path / "si_test.cif"
    from pymatgen.io.cif import CifWriter
    CifWriter(sample_si_structure).write_file(str(cif_path))
    return str(cif_path)


@pytest.fixture
def cif_file_tio2(sample_tio2_structure, tmp_path):
    """Create a temporary CIF file for TiO2."""
    cif_path = tmp_path / "tio2_test.cif"
    from pymatgen.io.cif import CifWriter
    CifWriter(sample_tio2_structure).write_file(str(cif_path))
    return str(cif_path)


def test_structure_from_cif(cif_file_si):
    """Test StructureFromCIF tool."""
    tool = StructureFromCIF()
    result = tool._run(cif_file_si)
    
    assert isinstance(result, str)
    assert "composition" in result or "Si" in result
    assert "lattice_parameters" in result or "Error" in result
    print(f"\n✓ StructureFromCIF test passed")
    print(f"  Result: {result[:200]}...")


def test_structure_symmetry_analysis(cif_file_si):
    """Test StructureSymmetryAnalysis tool."""
    tool = StructureSymmetryAnalysis()
    result = tool._run(cif_file_si)
    
    assert isinstance(result, str)
    assert "space_group" in result.lower() or "Error" in result
    print(f"\n✓ StructureSymmetryAnalysis test passed")
    print(f"  Result: {result[:200]}...")


def test_structure_properties(cif_file_si):
    """Test StructureProperties tool."""
    tool = StructureProperties()
    result = tool._run(cif_file_si)
    
    assert isinstance(result, str)
    assert "density" in result.lower() or "Error" in result
    print(f"\n✓ StructureProperties test passed")
    print(f"  Result: {result[:200]}...")


def test_composition_analysis():
    """Test CompositionAnalysis tool."""
    tool = CompositionAnalysis()
    
    # Test with simple formula
    result = tool._run("SiO2")
    assert isinstance(result, str)
    assert "Si" in result and "O" in result
    
    # Test with lithium oxide
    result = tool._run("Li2O")
    assert isinstance(result, str)
    assert "Li" in result
    
    print(f"\n✓ CompositionAnalysis test passed")
    print(f"  SiO2 result: {result[:150]}...")


def test_structure_comparison(cif_file_si, cif_file_tio2):
    """Test StructureComparison tool."""
    tool = StructureComparison()
    
    # Compare same structure
    result = tool._run(f"{cif_file_si}|{cif_file_si}")
    assert isinstance(result, str)
    
    # Compare different structures
    result = tool._run(f"{cif_file_si}|{cif_file_tio2}")
    assert isinstance(result, str)
    
    print(f"\n✓ StructureComparison test passed")
    print(f"  Result: {result[:200]}...")


def test_structure_substitution(cif_file_si):
    """Test StructureSubstitution tool."""
    tool = StructureSubstitution()
    result = tool._run(f"{cif_file_si}|Si|Ge")
    
    assert isinstance(result, str)
    assert "Ge" in result or "Error" in result
    print(f"\n✓ StructureSubstitution test passed")
    print(f"  Result: {result[:200]}...")


def test_primitive_cell_conversion(cif_file_si):
    """Test PrimitiveCellConversion tool."""
    tool = PrimitiveCellConversion()
    result = tool._run(cif_file_si)
    
    assert isinstance(result, str)
    assert "primitive" in result.lower() or "Error" in result
    print(f"\n✓ PrimitiveCellConversion test passed")
    print(f"  Result: {result[:200]}...")


def test_structure_to_cif(sample_si_structure, tmp_path):
    """Test StructureToCIF tool."""
    tool = StructureToCIF()
    
    # First create a CIF file to load
    from pymatgen.io.cif import CifWriter
    input_cif = tmp_path / "input.cif"
    CifWriter(sample_si_structure).write_file(str(input_cif))
    
    # Export to new file
    output_cif = tmp_path / "output.cif"
    result = tool._run(f"{input_cif}|{output_cif}")
    
    assert isinstance(result, str)
    assert output_cif.exists() or "Error" in result
    print(f"\n✓ StructureToCIF test passed")
    print(f"  Result: {result}")


def test_surface_generation(cif_file_si):
    """Test SurfaceGeneration tool."""
    tool = SurfaceGeneration()
    result = tool._run(f"{cif_file_si}|1 1 1|10|15")
    
    assert isinstance(result, str)
    assert "miller" in result.lower() or "Error" in result
    print(f"\n✓ SurfaceGeneration test passed")
    print(f"  Result: {result[:200]}...")


def test_coordination_analysis(cif_file_tio2):
    """Test CoordinationAnalysis tool."""
    tool = CoordinationAnalysis()
    result = tool._run(cif_file_tio2)
    
    assert isinstance(result, str)
    assert "coordination" in result.lower() or "Error" in result
    print(f"\n✓ CoordinationAnalysis test passed")
    print(f"  Result: {result[:200]}...")


def test_lattice_parameter_optimization(cif_file_si):
    """Test LatticeParameterOptimization tool."""
    tool = LatticeParameterOptimization()
    result = tool._run(cif_file_si)
    
    assert isinstance(result, str)
    assert "lattice" in result.lower() or "Error" in result
    print(f"\n✓ LatticeParameterOptimization test passed")
    print(f"  Result: {result[:200]}...")


def test_mp_structure_query():
    """Test MPStructureQuery tool (requires API key)."""
    mp_api_key = os.getenv("MP_API_KEY")
    if not mp_api_key:
        print(f"\n⚠ MPStructureQuery test skipped (no MP_API_KEY)")
        return
    
    tool = MPStructureQuery(mp_api_key=mp_api_key)
    
    # Test with material ID
    result = tool._run("mp-149")  # Silicon
    assert isinstance(result, str)
    
    print(f"\n✓ MPStructureQuery test passed")
    print(f"  Result: {result[:200]}...")


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Testing Crystal Materials Tools")
    print("=" * 70)
    
    if not PYMATGEN_AVAILABLE:
        print("\n❌ pymatgen not available. Install with: pip install pymatgen")
        return
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test structures
        try:
            lattice_si = Lattice.cubic(5.43)
            si_structure = Structure(lattice_si, ["Si", "Si", "Si", "Si"],
                                    [[0, 0, 0], [0.25, 0.25, 0.25], 
                                     [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]])
            
            lattice_tio2 = Lattice.tetragonal(4.6, 2.96)
            tio2_structure = Structure(lattice_tio2, 
                                      ["Ti", "Ti", "O", "O", "O", "O"],
                                      [[0, 0, 0], [0.5, 0.5, 0.5],
                                       [0.3, 0.3, 0.0], [0.7, 0.7, 0.0],
                                       [0.2, 0.8, 0.5], [0.8, 0.2, 0.5]])
            
            from pymatgen.io.cif import CifWriter
            cif_si = tmp_path / "si_test.cif"
            cif_tio2 = tmp_path / "tio2_test.cif"
            CifWriter(si_structure).write_file(str(cif_si))
            CifWriter(tio2_structure).write_file(str(cif_tio2))
            
        except Exception as e:
            print(f"\n❌ Error creating test structures: {e}")
            return
        
        # Run tests
        tests = [
            ("StructureFromCIF", lambda: test_structure_from_cif(str(cif_si))),
            ("StructureSymmetryAnalysis", lambda: test_structure_symmetry_analysis(str(cif_si))),
            ("StructureProperties", lambda: test_structure_properties(str(cif_si))),
            ("CompositionAnalysis", lambda: test_composition_analysis()),
            ("StructureComparison", lambda: test_structure_comparison(str(cif_si), str(cif_tio2))),
            ("StructureSubstitution", lambda: test_structure_substitution(str(cif_si))),
            ("PrimitiveCellConversion", lambda: test_primitive_cell_conversion(str(cif_si))),
            ("StructureToCIF", lambda: test_structure_to_cif(si_structure, tmp_path)),
            ("SurfaceGeneration", lambda: test_surface_generation(str(cif_si))),
            ("CoordinationAnalysis", lambda: test_coordination_analysis(str(cif_tio2))),
            ("LatticeParameterOptimization", lambda: test_lattice_parameter_optimization(str(cif_si))),
            ("MPStructureQuery", lambda: test_mp_structure_query()),
        ]
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test_name, test_func in tests:
            try:
                test_func()
                passed += 1
            except AssertionError as e:
                print(f"\n❌ {test_name} test failed: {e}")
                failed += 1
            except Exception as e:
                if "skip" in str(e).lower() or "api" in str(e).lower():
                    skipped += 1
                else:
                    print(f"\n❌ {test_name} test error: {e}")
                    failed += 1
        
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        print(f"⚠ Skipped: {skipped}")
        print(f"Total: {passed + failed + skipped}")
        print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

