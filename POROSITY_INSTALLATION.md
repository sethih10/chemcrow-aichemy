## Porosity Tools - Installation & Setup Guide

### 📋 Prerequisites Check

Before using the porosity tools, ensure you have:

```bash
# Check Python version (3.7+)
python --version

# Check ChemCrow is installed
python -c "import chemcrow; print('ChemCrow installed')"

# Check LangChain
python -c "from langchain.tools import BaseTool; print('LangChain ready')"
```

### ⚙️ Installation Steps

#### Step 1: Update ChemCrow (Already Done ✓)

The porosity tools are already integrated:

```bash
cd /scratch/work/sethih1/Crow/chemcrow-aichemy

# Verify installation
python -c "from chemcrow.tools.porosity import PorosityCalculator; print('✓ Tools available')"
```

#### Step 2: Install Zeo++ (REQUIRED for actual calculations)

**Option A: Using Conda (Recommended)**
```bash
conda install -c conda-forge zeo++
```

**Option B: Using Mamba (Faster)**
```bash
mamba install -c conda-forge zeo++
```

**Option C: From Source**
1. Visit http://zeo.chem.cmu.edu/
2. Download the source code
3. Follow compilation instructions
4. Add `bin/` directory to PATH

#### Step 3: Verify Zeo++ Installation

```bash
# Check Zeo++ is in PATH
which network

# If not found, add to PATH:
export PATH="/path/to/zeopp/bin:$PATH"

# Or add permanently to ~/.bashrc:
echo 'export PATH="/path/to/zeopp/bin:$PATH"' >> ~/.bashrc
```

### ✅ Verification

Run the test suite:

```bash
cd /scratch/work/sethih1/Crow/chemcrow-aichemy
python test_porosity_tool.py
```

Expected output:
```
✅ All tests completed!
```

### 🚀 Quick Usage Test

```python
from chemcrow.tools.porosity import PorosityCalculator

# This will work immediately (no Zeo++ needed for this check)
calc = PorosityCalculator()
print(f"Zeo++ status: {calc.zeopp_path}")

# For actual calculations, Zeo++ must be installed
if calc.zeopp_path:
    print("✓ Ready for calculations")
else:
    print("⚠ Install Zeo++ for calculations: conda install -c conda-forge zeo++")
```

### 📦 Package Dependencies

The porosity tools use:

**Python Standard Library (Already Available)**
- `subprocess` - For running Zeo++
- `tempfile` - For temporary files
- `os` - File operations
- `typing` - Type hints

**ChemCrow Dependencies (Already Available)**
- `langchain.tools.BaseTool` - Tool framework
- `pydantic.Field` - Data validation

**External Required**
- `Zeo++` - Porous material analysis (conda-installable)

### 🔧 Troubleshooting

#### Problem: "ModuleNotFoundError: No module named 'chemcrow.tools.porosity'"

**Solution:**
```bash
cd /scratch/work/sethih1/Crow/chemcrow-aichemy
pip install -e .
```

#### Problem: "zeopp_path not found after installation"

**Solution 1: Verify installation**
```bash
which network
# Should return a path like: /home/user/miniconda3/bin/network
```

**Solution 2: Check PATH**
```bash
echo $PATH
# Should include the conda/bin directory
```

**Solution 3: Reinstall with verbose output**
```bash
conda install -c conda-forge zeo++ -v
```

#### Problem: "Error running Zeo++: [error message]"

**Solution:**
1. Verify CIF file format with `validate_cif_file()`
2. Check file permissions: `chmod 644 your_file.cif`
3. Try with absolute path: `/absolute/path/to/file.cif`

### 📚 Documentation Files

Located in `/scratch/work/sethih1/Crow/chemcrow-aichemy/`:

| File | Purpose |
|------|---------|
| `POROSITY_TOOLS.md` | Complete documentation |
| `POROSITY_QUICK_REFERENCE.md` | Quick reference guide |
| `POROSITY_TOOLS_SUMMARY.md` | Feature overview |
| `examples_porosity_tools.py` | Usage examples |
| `test_porosity_tool.py` | Test suite |

### 🎯 Next Steps

1. **Get a CIF file:**
   - CoRE MOF Database: https://core.chem.cmu.edu/
   - CCDC: https://www.ccdc.cam.ac.uk/
   - Your own structure

2. **Run your first analysis:**
   ```python
   from chemcrow.tools.porosity import PorosityCalculator
   
   calc = PorosityCalculator()
   result = calc._run("your_structure.cif")
   print(result)
   ```

3. **Use in ChemCrow agent:**
   ```python
   from chemcrow.agents import ChemCrow
   
   agent = ChemCrow(model="gpt-4")
   result = agent.run("Analyze the porosity of MOF-5.cif")
   ```

### 📞 Support

1. Check test output: `python test_porosity_tool.py`
2. Review documentation: `POROSITY_TOOLS.md`
3. Check examples: `examples_porosity_tools.py`
4. Verify file format: `validate_cif_file("your_file.cif")`

### 🔗 Useful Links

- **Zeo++ Official**: http://zeo.chem.cmu.edu/
- **Zeo++ Installation**: http://zeo.chem.cmu.edu/mediawiki/index.php/Getting_started
- **CoRE MOF Database**: https://core.chem.cmu.edu/
- **CIF Format Specification**: https://www.iucr.org/resources/cif

### ✨ Environment Variables (Optional)

For advanced users:

```bash
# Set default Zeo++ path
export ZEOPP_PATH="/path/to/zeopp/bin/network"

# Set default probe radius (Ångströms)
export ZEOPP_PROBE_RADIUS="1.5"

# Set calculation timeout (seconds)
export ZEOPP_TIMEOUT="120"
```

### 📋 Installation Checklist

- [ ] Python 3.7+ installed
- [ ] ChemCrow installed and working
- [ ] Zeo++ installed: `conda install -c conda-forge zeo++`
- [ ] Zeo++ in PATH: `which network` returns a path
- [ ] Tests pass: `python test_porosity_tool.py`
- [ ] CIF files available (from CoRE or CCDC)
- [ ] Documentation reviewed

### 🎉 Ready to Use!

Once all items are checked, you're ready to:
- Calculate porosity of porous materials
- Analyze pore size distributions
- Use tools in ChemCrow agent
- Batch process multiple structures

Happy analyzing! 🔬

---

**Last Updated**: December 2025 | **For**: ChemCrow v1.0+
