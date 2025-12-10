"""
Porosity calculation tools using Zeo++.
Zeo++ is a software package for analyzing porous materials and calculating their properties.
"""

import subprocess
import tempfile
import os
from typing import Optional, Dict, Any
from langchain.tools import BaseTool
from pydantic import Field


class PorosityCalculator(BaseTool):
    """Calculate porosity metrics of porous materials using Zeo++."""
    
    name: str = "Porosity"
    description: str = (
        "Calculate porosity and related properties of porous materials. "
        "Input: CIF file path or structure file. "
        "Output: Porosity fraction, accessible volume, and surface area metrics."
    )
    zeopp_path: Optional[str] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        """Initialize the porosity calculator."""
        super().__init__(**kwargs)
        self.zeopp_path = self._find_zeopp()
    
    def _find_zeopp(self) -> Optional[str]:
        """Find Zeo++ executable in system PATH."""
        try:
            result = subprocess.run(
                ["which", "network"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def _run(self, cif_file_path: str) -> str:
        """
        Calculate porosity from a CIF (Crystallographic Information File).
        
        Args:
            cif_file_path: Path to the CIF file
            
        Returns:
            String containing porosity metrics
        """
        if not os.path.exists(cif_file_path):
            return f"Error: File not found at {cif_file_path}"
        
        if not self.zeopp_path:
            return (
                "Error: Zeo++ is not installed. "
                "Install it with: conda install -c conda-forge zeo++ "
                "or visit http://zeo.chem.cmu.edu/"
            )
        
        try:
            # Run network analysis to calculate porosity
            result = subprocess.run(
                [self.zeopp_path, "-ha", cif_file_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return f"Error running Zeo++: {result.stderr}"
            
            # Parse output
            metrics = self._parse_zeopp_output(result.stdout)
            return self._format_results(metrics, cif_file_path)
            
        except subprocess.TimeoutExpired:
            return "Error: Zeo++ calculation timed out (>60s)"
        except Exception as e:
            return f"Error calculating porosity: {str(e)}"
    
    def _parse_zeopp_output(self, output: str) -> Dict[str, Any]:
        """
        Parse Zeo++ output to extract porosity metrics.
        
        Args:
            output: Raw output from Zeo++ network analysis
            
        Returns:
            Dictionary with parsed metrics
        """
        metrics = {}
        
        for line in output.split('\n'):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Parse various Zeo++ output metrics
            if "Density" in line:
                try:
                    value = float(line.split()[-1])
                    metrics['density'] = value
                except (IndexError, ValueError):
                    pass
            elif "Volume" in line:
                try:
                    value = float(line.split()[-1])
                    metrics['volume'] = value
                except (IndexError, ValueError):
                    pass
            elif "Accessible" in line:
                try:
                    value = float(line.split()[-1])
                    metrics['accessible_volume'] = value
                except (IndexError, ValueError):
                    pass
        
        return metrics
    
    def _format_results(self, metrics: Dict[str, Any], cif_file: str) -> str:
        """
        Format porosity results for display.
        
        Args:
            metrics: Parsed metrics dictionary
            cif_file: Path to the CIF file analyzed
            
        Returns:
            Formatted results string
        """
        if not metrics:
            return (
                f"Warning: Zeo++ completed but no metrics were extracted from {cif_file}. "
                "The file may not be a valid MOF/porous material structure."
            )
        
        results = [f"Porosity Analysis for: {os.path.basename(cif_file)}", "=" * 50]
        
        if 'density' in metrics:
            results.append(f"Bulk Density: {metrics['density']:.4f} g/cm³")
        
        if 'volume' in metrics:
            results.append(f"Unit Cell Volume: {metrics['volume']:.2f} Ų")
        
        if 'accessible_volume' in metrics:
            av = metrics['accessible_volume']
            results.append(f"Accessible Volume: {av:.2f} Ų")
            
            # Calculate porosity fraction if volume available
            if 'volume' in metrics:
                porosity = av / metrics['volume']
                results.append(f"Porosity Fraction: {porosity:.4f} ({porosity*100:.2f}%)")
        
        results.append("\nNote: For more detailed analysis (surface area, pore size distribution),")
        results.append("run Zeo++ directly with additional parameters.")
        
        return '\n'.join(results)
    
    async def _arun(self, query: str) -> str:
        """Async version of the tool."""
        raise NotImplementedError("Use synchronous version")


class PoreSizeDistribution(BaseTool):
    """Calculate pore size distribution using Zeo++."""
    
    name: str = "PoreSizeDistribution"
    description: str = (
        "Calculate pore size distribution (largest included sphere, largest free sphere). "
        "Input: CIF file path. "
        "Output: Pore diameter statistics."
    )
    zeopp_path: Optional[str] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        """Initialize pore size calculator."""
        super().__init__(**kwargs)
        self.zeopp_path = self._find_zeopp()
    
    def _find_zeopp(self) -> Optional[str]:
        """Find Zeo++ executable in system PATH."""
        try:
            result = subprocess.run(
                ["which", "network"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def _run(self, cif_file_path: str) -> str:
        """
        Calculate pore size distribution.
        
        Args:
            cif_file_path: Path to the CIF file
            
        Returns:
            String containing pore size metrics
        """
        if not os.path.exists(cif_file_path):
            return f"Error: File not found at {cif_file_path}"
        
        if not self.zeopp_path:
            return (
                "Error: Zeo++ is not installed. "
                "Install it with: conda install -c conda-forge zeo++ "
                "or visit http://zeo.chem.cmu.edu/"
            )
        
        try:
            # Run Zeo++ with pore size analysis (-ha flag includes pore diameter)
            result = subprocess.run(
                [self.zeopp_path, "-ha", cif_file_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return f"Error running Zeo++: {result.stderr}"
            
            # Extract pore size information
            pore_info = self._extract_pore_sizes(result.stdout)
            return self._format_pore_results(pore_info, cif_file_path)
            
        except subprocess.TimeoutExpired:
            return "Error: Zeo++ calculation timed out (>60s)"
        except Exception as e:
            return f"Error calculating pore size: {str(e)}"
    
    def _extract_pore_sizes(self, output: str) -> Dict[str, Any]:
        """Extract pore size metrics from Zeo++ output."""
        pore_info = {}
        
        for line in output.split('\n'):
            line = line.strip()
            if "LCD" in line or "Largest" in line.upper():
                try:
                    value = float(line.split()[-1])
                    pore_info['largest_cavity_diameter'] = value
                except (IndexError, ValueError):
                    pass
            elif "LFS" in line or "Free" in line.upper():
                try:
                    value = float(line.split()[-1])
                    pore_info['largest_free_sphere'] = value
                except (IndexError, ValueError):
                    pass
        
        return pore_info
    
    def _format_pore_results(self, pore_info: Dict[str, Any], cif_file: str) -> str:
        """Format pore size results."""
        if not pore_info:
            return (
                f"Warning: Zeo++ completed but pore size data could not be extracted from {cif_file}. "
                "Run Zeo++ directly for detailed analysis."
            )
        
        results = [f"Pore Size Analysis for: {os.path.basename(cif_file)}", "=" * 50]
        
        if 'largest_cavity_diameter' in pore_info:
            lcd = pore_info['largest_cavity_diameter']
            results.append(f"Largest Cavity Diameter (LCD): {lcd:.2f} Å")
        
        if 'largest_free_sphere' in pore_info:
            lfs = pore_info['largest_free_sphere']
            results.append(f"Largest Free Sphere (LFS): {lfs:.2f} Å")
        
        results.append("\nThese metrics characterize the pore accessibility of the material.")
        
        return '\n'.join(results)
    
    async def _arun(self, query: str) -> str:
        """Async version of the tool."""
        raise NotImplementedError("Use synchronous version")
