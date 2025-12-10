# Porosity Tools - Quick Reference

## 🚀 Quick Start

### 1. Install Zeo++
```bash
conda install -c conda-forge zeo++
```

### 2. Import the tools
```python
from chemcrow.tools.porosity import PorosityCalculator, PoreSizeDistribution
from chemcrow.utils import validate_cif_file
```

### 3. Use the tools
```python
# Validate your CIF file
is_valid, msg = validate_cif_file("MOF-5.cif")

# Calculate porosity
calc = PorosityCalculator()
result = calc._run("MOF-5.cif")
print(result)

# Calculate pore sizes
pore_calc = PoreSizeDistribution()
result = pore_calc._run("MOF-5.cif")
print(result)
```

## 📊 Tool Outputs

### PorosityCalculator Output
```
Porosity Analysis for: MOF-5.cif
==================================================
Bulk Density: 0.4532 g/cm³
Unit Cell Volume: 8853.42 Ų
Accessible Volume: 4512.75 Ų
Porosity Fraction: 0.5098 (50.98%)
```

### PoreSizeDistribution Output
```
Pore Size Analysis for: MOF-5.cif
==================================================
Largest Cavity Diameter (LCD): 8.65 Å
Largest Free Sphere (LFS): 6.24 Å
```

## ✅ Quick Interpretation

| Metric | Good Value | Interpretation |
|--------|-----------|-----------------|
| Porosity > 0.5 | ✓ Excellent | Very high surface area |
| Porosity 0.3-0.5 | ✓ Good | Moderate accessibility |
| Porosity < 0.3 | ⚠ Lower | Dense material |
| LCD > 10 Å | ✓ Large pores | Gas storage capability |
| LFS < LCD | ⚠ Bottleneck | Kinetic constraints |

## 🔍 Common Issues & Solutions

### Issue: "Zeo++ is not installed"
```bash
# Solution:
conda install -c conda-forge zeo++
which network  # Verify installation
```

### Issue: "Invalid CIF file"
Check that your file:
- Starts with `data_` block
- Has crystallographic parameters (`_cell_length_*`, etc.)
- Has valid atomic coordinates

### Issue: "File not found"
Use absolute paths:
```python
import os
cif_path = os.path.abspath("MOF-5.cif")
result = calc._run(cif_path)
```

## 📝 File Formats

### Supported Input
- **.cif** files (Crystallographic Information Files)
- Must be valid crystal structures
- Typically from CCDC or CoRE databases

### Finding CIF Files
- **CoRE MOF Database**: https://core.chem.cmu.edu/ (5000+ MOFs)
- **CCDC**: https://www.ccdc.cam.ac.uk/ (>1 million structures)
- **ICSD**: https://icsd.products.fiz-karlsruhe.de/ (Inorganic crystals)

## 🛠️ Advanced Usage

### Direct Zeo++ Commands
```bash
network -ha MOF-5.cif              # Basic analysis
network -ha -r 1.5 MOF-5.cif      # High accuracy
network -bls MOF-5.cif             # Blocking spheres
network -sa 1.5 MOF-5.cif         # Surface area
```

### Batch Processing
```python
from pathlib import Path
from chemcrow.tools.porosity import PorosityCalculator

calc = PorosityCalculator()
for cif_file in Path(".").glob("*.cif"):
    result = calc._run(str(cif_file))
    print(f"{cif_file.name}: {result}")
```

### Integration with ChemCrow Agent
```python
from chemcrow.agents import ChemCrow

agent = ChemCrow(model="gpt-4", temp=0.1)
result = agent.run("Calculate porosity of MOF-5.cif and suggest applications")
print(result)
```

## 📚 Resources

- **Zeo++ Official**: http://zeo.chem.cmu.edu/
- **Zeo++ Paper**: Willems et al., Microporous and Mesoporous Materials, 2011
- **CIF Format Guide**: https://www.iucr.org/resources/cif
- **MOF References**: https://core.chem.cmu.edu/

## 🎯 Use Cases

| Application | Required Metric | Optimal Value |
|-------------|-----------------|---------------|
| CO2 Capture | Accessible Volume | High (>40%) |
| H2 Storage | Large Cavity | LCD > 12 Å |
| CH4 Storage | Porosity | >0.5 |
| Catalysis | Surface Area | >1000 m²/g |
| Separation | Pore Size | Matched to molecules |

## 📞 Support

For issues or feature requests:
1. Check CIF file validity with `validate_cif_file()`
2. Verify Zeo++ installation: `which network`
3. Review POROSITY_TOOLS.md for detailed documentation
4. Check test results: `python test_porosity_tool.py`

## 💡 Pro Tips

1. **Validate first**: Always validate CIF files before processing
2. **Use absolute paths**: Avoid relative path issues
3. **Check Zeo++**: Ensure `which network` returns a path
4. **Start simple**: Test with known MOF structures first
5. **Read output carefully**: Check for warnings in results

---

**Created**: December 2025 | **For**: ChemCrow v1.0+ | **Requires**: Zeo++ installed
