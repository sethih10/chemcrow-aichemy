"""Crystal materials tools using pymatgen for structure analysis and manipulation."""

import json
from typing import Optional

from langchain.tools import BaseTool

try:
    from pymatgen.core import Structure, Lattice, Composition
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.cif import CifParser, CifWriter

    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


class StructureFromCIF(BaseTool):
    """Load crystal structure from CIF file."""

    name = "StructureFromCIF"
    description = (
        "Load a crystal structure from a CIF file. "
        "Input: path to CIF file. "
        "Returns: structure information including lattice parameters, composition, and space group."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _run(self, cif_path: str) -> str:
        """Load structure from CIF file."""
        try:
            parser = CifParser(cif_path)
            structure = parser.get_structures()[0]

            sga = SpacegroupAnalyzer(structure)
            space_group = sga.get_space_group_symbol()

            info = {
                "composition": structure.composition.formula,
                "lattice_parameters": {
                    "a": structure.lattice.a,
                    "b": structure.lattice.b,
                    "c": structure.lattice.c,
                    "alpha": structure.lattice.alpha,
                    "beta": structure.lattice.beta,
                    "gamma": structure.lattice.gamma,
                },
                "volume": structure.volume,
                "density": structure.density,
                "num_sites": len(structure),
                "space_group": space_group,
            }

            return json.dumps(info)
        except Exception as e:
            return f"Error loading CIF file: {str(e)}"

    async def _arun(self, cif_path: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class StructureSymmetryAnalysis(BaseTool):
    """Analyze crystal structure symmetry properties."""

    name = "StructureSymmetryAnalysis"
    description = (
        "Analyze symmetry properties of a crystal structure. "
        "Input: path to CIF file or structure in JSON format. "
        "Returns: space group, point group, crystal system, and symmetry operations."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _load_structure(self, input_data: str):
        """Load structure from CIF path or JSON string."""
        try:
            # Try as JSON first
            data = json.loads(input_data)
            if "lattice" in data and "sites" in data:
                return Structure.from_dict(data)
        except:
            pass

        # Try as CIF path
        try:
            parser = CifParser(input_data)
            return parser.get_structures()[0]
        except:
            pass

        raise ValueError(f"Could not parse structure from: {input_data}")

    def _run(self, structure_input: str) -> str:
        """Analyze structure symmetry."""
        try:
            structure = self._load_structure(structure_input)
            sga = SpacegroupAnalyzer(structure)

            info = {
                "space_group_symbol": sga.get_space_group_symbol(),
                "space_group_number": sga.get_space_group_number(),
                "crystal_system": sga.get_crystal_system(),
                "point_group": sga.get_point_group_symbol(),
                "symmetry_operations": len(sga.get_symmetry_operations()),
                "is_primitive": sga.is_primitive(),
            }

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error analyzing symmetry: {str(e)}"

    async def _arun(self, structure_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class StructureProperties(BaseTool):
    """Calculate basic properties of a crystal structure."""

    name = "StructureProperties"
    description = (
        "Calculate physical and chemical properties of a crystal structure. "
        "Input: path to CIF file or structure JSON. "
        "Returns: density, volume, packing fraction, coordination numbers, and more."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _load_structure(self, input_data: str):
        """Load structure from CIF path or JSON string."""
        try:
            data = json.loads(input_data)
            if "lattice" in data and "sites" in data:
                return Structure.from_dict(data)
        except:
            pass

        try:
            parser = CifParser(input_data)
            return parser.get_structures()[0]
        except:
            pass

        raise ValueError(f"Could not parse structure from: {input_data}")

    def _run(self, structure_input: str) -> str:
        """Calculate structure properties."""
        try:
            structure = self._load_structure(structure_input)

            # Calculate average coordination
            from pymatgen.analysis.local_env import MinimumDistanceNN

            nn = MinimumDistanceNN()
            coord_numbers = []
            for i in range(len(structure)):
                coord_numbers.append(len(nn.get_nn(structure, i)))

            info = {
                "composition": structure.composition.formula,
                "formula_weight": structure.composition.weight,
                "density_g_cm3": structure.density,
                "volume_A3": structure.volume,
                "num_atoms": len(structure),
                "avg_coordination_number": (
                    sum(coord_numbers) / len(coord_numbers) if coord_numbers else 0
                ),
                "lattice_type": str(structure.lattice.type),
                "lattice_volume_A3": structure.lattice.volume,
            }

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error calculating properties: {str(e)}"

    async def _arun(self, structure_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class StructureComparison(BaseTool):
    """Compare two crystal structures for similarity."""

    name = "StructureComparison"
    description = (
        "Compare two crystal structures to determine if they are similar. "
        "Input: two CIF file paths or structure JSONs separated by '|'. "
        "Returns: comparison results including whether structures match and similarity metrics."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _load_structure(self, input_data: str):
        """Load structure from CIF path or JSON string."""
        try:
            data = json.loads(input_data)
            if "lattice" in data and "sites" in data:
                return Structure.from_dict(data)
        except:
            pass

        try:
            parser = CifParser(input_data)
            return parser.get_structures()[0]
        except:
            pass

        raise ValueError(f"Could not parse structure from: {input_data}")

    def _run(self, structures_input: str) -> str:
        """Compare two structures."""
        try:
            if "|" not in structures_input:
                return "Error: Please provide two structures separated by '|'"

            paths = [s.strip() for s in structures_input.split("|")]
            if len(paths) != 2:
                return "Error: Please provide exactly two structures separated by '|'"

            struct1 = self._load_structure(paths[0])
            struct2 = self._load_structure(paths[1])

            matcher = StructureMatcher()
            matches = matcher.fit(struct1, struct2)

            # Calculate composition similarity
            comp1 = struct1.composition
            comp2 = struct2.composition
            comp_similar = comp1.almost_equals(comp2, rtol=0.01)

            info = {
                "structures_match": matches,
                "compositions_similar": comp_similar,
                "structure1_formula": comp1.formula,
                "structure2_formula": comp2.formula,
                "structure1_volume": struct1.volume,
                "structure2_volume": struct2.volume,
            }

            if matches:
                info["matching_info"] = "Structures are structurally similar"
            else:
                info["matching_info"] = "Structures are different"

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error comparing structures: {str(e)}"

    async def _arun(self, structures_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class CompositionAnalysis(BaseTool):
    """Analyze chemical composition of materials."""

    name = "CompositionAnalysis"
    description = (
        "Analyze a chemical composition (e.g., 'Li2O', 'SiO2', 'Fe2O3'). "
        "Input: chemical formula as string. "
        "Returns: molecular weight, element fractions, oxidation states estimation."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _run(self, formula: str) -> str:
        """Analyze composition."""
        try:
            comp = Composition(formula)

            info = {
                "formula": comp.formula,
                "reduced_formula": comp.reduced_formula,
                "weight": comp.weight,
                "num_atoms": comp.num_atoms,
                "elements": [str(e) for e in comp.elements],
                "element_fractions": {
                    str(e): comp.get_atomic_fraction(e) for e in comp.elements
                },
                "element_mass_fractions": {
                    str(e): comp.get_wt_fraction(e) for e in comp.elements
                },
            }

            # Try to get oxidation states
            try:
                comp_with_oxi = comp.add_charges_from_oxi_state_guesses()
                info["estimated_oxidation_states"] = {
                    str(k): v for k, v in comp_with_oxi.oxi_state_guesses()[0].items()
                }
            except:
                info["estimated_oxidation_states"] = "Could not estimate"

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error analyzing composition: {str(e)}"

    async def _arun(self, formula: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class StructureSubstitution(BaseTool):
    """Substitute elements in a crystal structure."""

    name = "StructureSubstitution"
    description = (
        "Substitute one element with another in a crystal structure. "
        "Input: structure (CIF path or JSON), old element, new element, separated by '|'. "
        "Returns: new structure information."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _load_structure(self, input_data: str):
        """Load structure from CIF path or JSON string."""
        try:
            data = json.loads(input_data)
            if "lattice" in data and "sites" in data:
                return Structure.from_dict(data)
        except:
            pass

        try:
            parser = CifParser(input_data)
            return parser.get_structures()[0]
        except:
            pass

        raise ValueError(f"Could not parse structure from: {input_data}")

    def _run(self, substitution_input: str) -> str:
        """Substitute elements."""
        try:
            parts = [s.strip() for s in substitution_input.split("|")]
            if len(parts) != 3:
                return (
                    "Error: Input format should be: structure|old_element|new_element"
                )

            structure_input, old_elem, new_elem = parts
            structure = self._load_structure(structure_input)

            new_structure = structure.copy()
            new_structure.replace_species({old_elem: new_elem})

            info = {
                "original_composition": structure.composition.formula,
                "new_composition": new_structure.composition.formula,
                "substitution": f"{old_elem} -> {new_elem}",
                "new_lattice_parameters": {
                    "a": new_structure.lattice.a,
                    "b": new_structure.lattice.b,
                    "c": new_structure.lattice.c,
                },
                "new_volume": new_structure.volume,
                "new_density": new_structure.density,
            }

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error performing substitution: {str(e)}"

    async def _arun(self, substitution_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class PrimitiveCellConversion(BaseTool):
    """Convert structure to primitive cell."""

    name = "PrimitiveCellConversion"
    description = (
        "Convert a crystal structure to its primitive cell representation. "
        "Input: path to CIF file or structure JSON. "
        "Returns: primitive cell structure information."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _load_structure(self, input_data: str):
        """Load structure from CIF path or JSON string."""
        try:
            data = json.loads(input_data)
            if "lattice" in data and "sites" in data:
                return Structure.from_dict(data)
        except:
            pass

        try:
            parser = CifParser(input_data)
            return parser.get_structures()[0]
        except:
            pass

        raise ValueError(f"Could not parse structure from: {input_data}")

    def _run(self, structure_input: str) -> str:
        """Convert to primitive cell."""
        try:
            structure = self._load_structure(structure_input)
            sga = SpacegroupAnalyzer(structure)
            primitive = sga.get_primitive_standard_structure()

            info = {
                "original_num_sites": len(structure),
                "primitive_num_sites": len(primitive),
                "reduction_ratio": len(primitive) / len(structure),
                "primitive_composition": primitive.composition.formula,
                "primitive_lattice_parameters": {
                    "a": primitive.lattice.a,
                    "b": primitive.lattice.b,
                    "c": primitive.lattice.c,
                    "alpha": primitive.lattice.alpha,
                    "beta": primitive.lattice.beta,
                    "gamma": primitive.lattice.gamma,
                },
                "primitive_volume": primitive.volume,
            }

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error converting to primitive cell: {str(e)}"

    async def _arun(self, structure_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class StructureToCIF(BaseTool):
    """Export structure to CIF format."""

    name = "StructureToCIF"
    description = (
        "Export a crystal structure to CIF format and save to file. "
        "Input: structure (CIF path or JSON) and output path, separated by '|'. "
        "Returns: confirmation message with file path."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _load_structure(self, input_data: str):
        """Load structure from CIF path or JSON string."""
        try:
            data = json.loads(input_data)
            if "lattice" in data and "sites" in data:
                return Structure.from_dict(data)
        except:
            pass

        try:
            parser = CifParser(input_data)
            return parser.get_structures()[0]
        except:
            pass

        raise ValueError(f"Could not parse structure from: {input_data}")

    def _run(self, export_input: str) -> str:
        """Export structure to CIF."""
        try:
            parts = [s.strip() for s in export_input.split("|")]
            if len(parts) != 2:
                return "Error: Input format should be: structure|output_path.cif"

            structure_input, output_path = parts
            structure = self._load_structure(structure_input)

            CifWriter(structure).write_file(output_path)

            return f"Structure exported successfully to {output_path}"
        except Exception as e:
            return f"Error exporting to CIF: {str(e)}"

    async def _arun(self, export_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


# Materials Project integration (requires API key)
class MPStructureQuery(BaseTool):
    """Query Materials Project database for crystal structures."""

    name = "MPStructureQuery"
    description = (
        "Query the Materials Project database for crystal structures. "
        "Input: material ID (e.g., 'mp-149' for silicon) or chemical formula. "
        "Requires MP_API_KEY environment variable. "
        "Returns: structure information from Materials Project."
    )

    def __init__(self, mp_api_key: Optional[str] = None):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

        self.mp_api_key = mp_api_key

        try:
            from mp_api.client import MPRester

            self.MPRester = MPRester
        except ImportError:
            self.MPRester = None

    def _run(self, query: str) -> str:
        """Query Materials Project."""
        if self.MPRester is None:
            return (
                "Error: mp-api package not installed. Install with: pip install mp-api"
            )

        if not self.mp_api_key:
            import os

            self.mp_api_key = os.getenv("MP_API_KEY")

        if not self.mp_api_key:
            return "Error: MP_API_KEY not provided. Set as environment variable or pass to constructor."

        try:
            with self.MPRester(self.mp_api_key) as mpr:
                # Try as material ID first
                try:
                    structure = mpr.get_structure_by_material_id(query)
                except:
                    # Try as formula
                    docs = mpr.summary.search(
                        formula=query,
                        fields=["material_id", "formula_pretty", "structure"],
                    )
                    if not docs:
                        return f"No materials found for query: {query}"
                    structure = docs[0].structure

                sga = SpacegroupAnalyzer(structure)

                info = {
                    "material_id": query if query.startswith("mp-") else "N/A",
                    "composition": structure.composition.formula,
                    "space_group": sga.get_space_group_symbol(),
                    "lattice_parameters": {
                        "a": structure.lattice.a,
                        "b": structure.lattice.b,
                        "c": structure.lattice.c,
                    },
                    "volume": structure.volume,
                    "density": structure.density,
                }

                return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error querying Materials Project: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class SurfaceGeneration(BaseTool):
    """Generate surface slabs from bulk crystal structures."""

    name = "SurfaceGeneration"
    description = (
        "Generate surface slabs from a bulk crystal structure. "
        "Input: structure (CIF path or JSON), miller indices (h k l), slab thickness, vacuum thickness, separated by '|'. "
        "Example: 'structure.cif|1 1 1|10|15' for (111) surface with 10 Angstrom slab and 15 Angstrom vacuum. "
        "Returns: surface slab structure information."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _load_structure(self, input_data: str):
        """Load structure from CIF path or JSON string."""
        try:
            data = json.loads(input_data)
            if "lattice" in data and "sites" in data:
                return Structure.from_dict(data)
        except:
            pass

        try:
            parser = CifParser(input_data)
            return parser.get_structures()[0]
        except:
            pass

        raise ValueError(f"Could not parse structure from: {input_data}")

    def _run(self, surface_input: str) -> str:
        """Generate surface slab."""
        try:
            from pymatgen.core.surface import SlabGenerator

            parts = [s.strip() for s in surface_input.split("|")]
            if len(parts) < 3:
                return "Error: Input format should be: structure|h k l|slab_thickness [vacuum_thickness]"

            structure_input, miller_str, slab_thickness = (
                parts[0],
                parts[1],
                float(parts[2]),
            )
            vacuum_thickness = float(parts[3]) if len(parts) > 3 else 15.0

            structure = self._load_structure(structure_input)
            miller_indices = tuple(map(int, miller_str.split()))

            slabgen = SlabGenerator(
                structure,
                miller_indices,
                min_slab_size=slab_thickness,
                min_vacuum_size=vacuum_thickness,
                center_slab=True,
            )
            slab = slabgen.get_slabs()[0]  # Get first termination

            info = {
                "miller_indices": miller_indices,
                "slab_thickness_A": slab_thickness,
                "vacuum_thickness_A": vacuum_thickness,
                "surface_area_A2": slab.surface_area,
                "num_sites_in_slab": len(slab),
                "slab_composition": slab.composition.formula,
                "slab_lattice_parameters": {
                    "a": slab.lattice.a,
                    "b": slab.lattice.b,
                    "c": slab.lattice.c,
                },
            }

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error generating surface: {str(e)}"

    async def _arun(self, surface_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class CoordinationAnalysis(BaseTool):
    """Analyze coordination environments in crystal structures."""

    name = "CoordinationAnalysis"
    description = (
        "Analyze coordination numbers and environments for each site in a crystal structure. "
        "Input: structure (CIF path or JSON) and optional cutoff radius in Angstrom. "
        "If cutoff not provided, uses nearest neighbor algorithm. "
        "Returns: coordination numbers and nearest neighbor information for each site."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _load_structure(self, input_data: str):
        """Load structure from CIF path or JSON string."""
        try:
            data = json.loads(input_data)
            if "lattice" in data and "sites" in data:
                return Structure.from_dict(data)
        except:
            pass

        try:
            parser = CifParser(input_data)
            return parser.get_structures()[0]
        except:
            pass

        raise ValueError(f"Could not parse structure from: {input_data}")

    def _run(self, coordination_input: str) -> str:
        """Analyze coordination."""
        try:
            from pymatgen.analysis.local_env import MinimumDistanceNN, VoronoiNN
            from collections import Counter

            parts = [s.strip() for s in coordination_input.split("|")]
            structure_input = parts[0]
            cutoff = float(parts[1]) if len(parts) > 1 else None

            structure = self._load_structure(structure_input)

            if cutoff:
                from pymatgen.analysis.local_env import CrystalNN

                nn = CrystalNN(search_cutoff=cutoff)
            else:
                nn = MinimumDistanceNN()

            coord_info = []
            all_coord_numbers = []

            for i, site in enumerate(structure):
                neighbors = nn.get_nn(structure, i)
                coord_num = len(neighbors)
                all_coord_numbers.append(coord_num)

                # Count neighbor species
                neighbor_species = Counter([n.specie.symbol for n in neighbors])

                site_info = {
                    "site_index": i,
                    "species": str(site.specie),
                    "coordination_number": coord_num,
                    "neighbor_species": dict(neighbor_species),
                    "avg_neighbor_distance_A": (
                        sum(site.distance(n) for n in neighbors) / coord_num
                        if neighbors
                        else 0
                    ),
                }
                coord_info.append(site_info)

            info = {
                "structure_composition": structure.composition.formula,
                "average_coordination_number": sum(all_coord_numbers)
                / len(all_coord_numbers),
                "coordination_distribution": dict(Counter(all_coord_numbers)),
                "site_details": coord_info,
            }

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error analyzing coordination: {str(e)}"

    async def _arun(self, coordination_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class LatticeParameterOptimization(BaseTool):
    """Analyze and suggest optimal lattice parameters."""

    name = "LatticeParameterOptimization"
    description = (
        "Analyze lattice parameters and provide information for optimization. "
        "Input: structure (CIF path or JSON). "
        "Returns: current lattice parameters, volume per atom, and suggestions for optimization."
    )

    def __init__(self):
        super().__init__()
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required for crystal materials tools. Install with: pip install pymatgen"
            )

    def _load_structure(self, input_data: str):
        """Load structure from CIF path or JSON string."""
        try:
            data = json.loads(input_data)
            if "lattice" in data and "sites" in data:
                return Structure.from_dict(data)
        except:
            pass

        try:
            parser = CifParser(input_data)
            return parser.get_structures()[0]
        except:
            pass

        raise ValueError(f"Could not parse structure from: {input_data}")

    def _run(self, structure_input: str) -> str:
        """Analyze lattice parameters."""
        try:
            structure = self._load_structure(structure_input)

            lattice = structure.lattice
            num_atoms = len(structure)

            info = {
                "lattice_type": lattice.type,
                "lattice_parameters_A": {
                    "a": lattice.a,
                    "b": lattice.b,
                    "c": lattice.c,
                    "alpha": lattice.alpha,
                    "beta": lattice.beta,
                    "gamma": lattice.gamma,
                },
                "volume_A3": lattice.volume,
                "volume_per_atom_A3": lattice.volume / num_atoms,
                "density_g_cm3": structure.density,
                "is_primitive": "Check with PrimitiveCellConversion tool",
            }

            # Check if lattice is standard
            try:
                sga = SpacegroupAnalyzer(structure)
                conventional = sga.get_conventional_standard_structure()
                if abs(conventional.lattice.volume - lattice.volume) < 0.01:
                    info["note"] = (
                        "Structure may benefit from conversion to conventional cell"
                    )
            except:
                pass

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error analyzing lattice parameters: {str(e)}"

    async def _arun(self, structure_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


__all__ = [
    "StructureFromCIF",
    "StructureSymmetryAnalysis",
    "StructureProperties",
    "StructureComparison",
    "CompositionAnalysis",
    "StructureSubstitution",
    "PrimitiveCellConversion",
    "StructureToCIF",
    "MPStructureQuery",
    "SurfaceGeneration",
    "CoordinationAnalysis",
    "LatticeParameterOptimization",
]
