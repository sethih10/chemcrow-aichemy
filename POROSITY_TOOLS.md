# Porosity Calculation Tools

This module provides tools for calculating the porosity and pore size characteristics of porous materials (MOFs, zeolites, etc.) using **Zeo++**.

## Overview

Two main tools are available:

### 1. **Porosity** Tool
Calculates:
- Bulk density
- Unit cell volume
- Accessible volume
- **Porosity fraction** (as percentage and decimal)

### 2. **PoreSizeDistribution** Tool
Calculates:
- Largest Cavity Diameter (LCD)
- Largest Free Sphere (LFS)
- Other pore metrics

## Installation

Before using these tools, you need to install **Zeo++**:

### Option A: Using Conda (Recommended)
```bash
conda install -c conda-forge zeo++
```

### Option B: From Source
Visit http://zeo.chem.cmu.edu/ and follow the installation instructions.

### Verify Installation
```bash
which network
```

If this returns a path, Zeo++ is installed and ready to use.

## Usage

### Basic Usage with ChemCrow Agent

```python
from chemcrow.agents import ChemCrow

# Initialize ChemCrow agent
chem_model = ChemCrow(model="gpt-4", temp=0.1)

# Query with porosity tool
result = chem_model.run("Calculate the porosity of the MOF-5 structure in /path/to/MOF-5.cif")
print(result)
```

### Direct Tool Usage

```python
from chemcrow.tools.porosity import PorosityCalculator, PoreSizeDistribution

# Create calculator
calc = PorosityCalculator()

# Calculate porosity
result = calc._run("/path/to/structure.cif")
print(result)

# Calculate pore size distribution
pore_calc = PoreSizeDistribution()
pore_result = pore_calc._run("/path/to/structure.cif")
print(pore_result)
```

### CIF File Validation

Before submitting a CIF file to the porosity calculator, you can validate it:

```python
from chemcrow.utils import validate_cif_file

is_valid, message = validate_cif_file("path/to/file.cif")
print(f"Valid: {is_valid}")
print(f"Message: {message}")
```

## Input Format: CIF Files

**CIF** (Crystallographic Information File) is the standard format for storing crystal structure data.

### Required Structure:
```cif
data_MOF_NAME
_cell_length_a    20.711
_cell_length_b    20.711
_cell_length_c    20.711
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90

loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_type_symbol
Zn1   0.0000 0.0000 0.0000 Zn
O1    0.0958 0.0958 0.0000 O
...
```

### Finding CIF Files:
- **CCDC** (Cambridge Crystallographic Data Centre): https://www.ccdc.cam.ac.uk/
- **CoRE MOF Database**: https://core.chem.cmu.edu/
- **ICSD** (Inorganic Crystal Structure Database): https://icsd.products.fiz-karlsruhe.de/

## Output Interpretation

### Porosity Results Example:
```
Porosity Analysis for: MOF-5.cif
==================================================
Bulk Density: 0.4532 g/cm³
Unit Cell Volume: 8853.42 Ų
Accessible Volume: 4512.75 Ų
Porosity Fraction: 0.5098 (50.98%)
```

**What it means:**
- **Porosity Fraction > 0.5**: Very high porosity (excellent for gas storage)
- **Porosity Fraction 0.3-0.5**: Moderate porosity
- **Porosity Fraction < 0.3**: Lower porosity

### Pore Size Distribution Results Example:
```
Pore Size Analysis for: MOF-5.cif
==================================================
Largest Cavity Diameter (LCD): 8.65 Å
Largest Free Sphere (LFS): 6.24 Å
```

**What it means:**
- **LCD**: Maximum sphere that can fit in the largest cavity
- **LFS**: Maximum sphere that can pass through the pore openings
- **LFS < LCD**: Material has bottleneck pores (kinetic constraints)

## Advanced Zeo++ Options

For more detailed analysis, you can run Zeo++ directly from the command line:

```bash
# Atomic structure analysis
network -ha MOF-5.cif

# High accuracy mode
network -ha -r 1.2 MOF-5.cif

# Generate blocking spheres (shows accessible volume)
network -bls MOF-5.cif

# Full contact surface
network -sa 1.5 MOF-5.cif

# Extended analysis
network -eva MOF-5.cif
```

## Common Parameters:

| Flag | Description |
|------|-------------|
| `-ha` | High accuracy atomic analysis |
| `-r` | Probe radius (in Ångströms) |
| `-bls` | Generate blocking spheres |
| `-sa` | Surface area calculation (radius) |
| `-eva` | Evacuation analysis |
| `-p` | Pore limit diameter |

## Error Handling

The tools include comprehensive error handling:

```python
result = calc._run("invalid_file.cif")
# Returns: "Error: File not found at invalid_file.cif"

result = calc._run("file.xyz")
# Returns: "File must have .cif extension"
```

## Performance Notes

- **Calculation time**: Usually 1-60 seconds depending on structure complexity
- **Timeout**: 60 seconds maximum per calculation
- **Memory**: Minimal requirements (~100 MB)

## Supported Material Types

These tools work well with:
- **Metal-Organic Frameworks (MOFs)**
- **Zeolites**
- **Porous Polymers**
- **Covalent Organic Frameworks (COFs)**
- **Any crystalline porous material**

## References

- **Zeo++ Paper**: Willems et al., Microporous and Mesoporous Materials, 2011
  - DOI: 10.1016/j.micromeso.2011.04.017
- **Official Zeo++ Site**: http://zeo.chem.cmu.edu/
- **CIF Format**: https://www.iucr.org/resources/cif

## Troubleshooting

### "Zeo++ is not installed"
```bash
conda install -c conda-forge zeo++
# or
conda install -c conda-forge zeo++ -y
```

### "Invalid CIF file" errors
- Check that file starts with `data_` block
- Verify crystallographic parameters are present
- Use online CIF validators: https://www.ccdc.cam.ac.uk/Community/inputfiles/

### "Calculation timed out"
- Structure may be too complex
- Try reducing with higher symmetry
- Or run Zeo++ directly with specific parameters

## Contributing

To add more analysis tools or improve accuracy:

1. Check Zeo++ command-line options
2. Add parsing logic in `_parse_zeopp_output()`
3. Format results with `_format_results()`
4. Test with known MOF structures

## License

These tools wrap Zeo++, which is distributed under the GNU Affero General Public License.
