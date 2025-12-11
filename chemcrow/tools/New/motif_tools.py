import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain.tools import BaseTool
from pymatgen.core import Structure
from pymatgen.analysis.local_env import MinimumDistanceNN

# Optional: handle NumPy types when they appear
try:
    import numpy as np
except ImportError:
    np = None


def _json_default(o):
    """
    Helper for json.dumps to convert NumPy types to plain Python types.
    """
    if np is not None:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    # Fallback: just stringify anything unknown
    return str(o)


# =========================
# Motif representation
# =========================

@dataclass
class MotifPattern:
    """
    Simple chemical motif: central species + neighbor species counts + max distance.

    Example JSON entry:
    {
      "name": "TiO6_oct",
      "central_species": "Ti",
      "neighbor_species_counts": {"O": 6},
      "max_distance": 2.3
    }
    """
    name: str
    central_species: str
    neighbor_species_counts: Dict[str, int]
    max_distance: float
    extra: Dict[str, Any] = field(default_factory=dict)


def _load_motif_library_from_file(path: str | Path) -> List[MotifPattern]:
    data = json.loads(Path(path).read_text())
    return [MotifPattern(**entry) for entry in data]


def _load_motif_library(
    motifs: Optional[Sequence[Dict]] = None,
    motif_library_path: Optional[str] = None,
) -> List[MotifPattern]:
    """
    Load motif patterns either from an inline list (motifs)
    or from a JSON file (motif_library_path).
    """
    if motifs is not None:
        return [MotifPattern(**m) for m in motifs]
    if motif_library_path is not None:
        return _load_motif_library_from_file(motif_library_path)
    raise ValueError("No motif definitions provided: supply 'motifs' or 'motif_library_path'.")


# =========================
# Core motif logic
# =========================

def _get_neighbor_info(
    structure: Structure,
    central_index: int,
    max_distance: float,
    nn_strategy=None,
):
    if nn_strategy is None:
        nn_strategy = MinimumDistanceNN()
    neighs = nn_strategy.get_nn_info(structure, central_index)

    neighbors = []
    central_site = structure[central_index]
    for info in neighs:
        j = info["site_index"]
        dist = central_site.distance(structure[j])
        if dist <= max_distance:
            species_j = (
                structure[j].specie.symbol
                if hasattr(structure[j], "specie")
                else structure[j].species_string
            )
            neighbors.append((j, species_j, dist))

    return neighbors


def _match_pattern(
    structure: Structure,
    central_index: int,
    pattern: MotifPattern,
) -> Optional[Dict]:
    # Check central species first
    central_site = structure[central_index]
    central_species = (
        central_site.specie.symbol
        if hasattr(central_site, "specie")
        else central_site.species_string
    )
    if central_species != pattern.central_species:
        return None

    neighbors = _get_neighbor_info(structure, central_index, pattern.max_distance)
    if not neighbors:
        return None

    # Count neighbors by species
    counts: Dict[str, int] = {}
    for _, species, _ in neighbors:
        counts[species] = counts.get(species, 0) + 1

    # Fail fast if we don't have enough of any required species
    for sp, req in pattern.neighbor_species_counts.items():
        if counts.get(sp, 0) < req:
            return None

    # Choose closest neighbors for each species
    chosen_neighbors: List[int] = []
    for sp, req in pattern.neighbor_species_counts.items():
        cand = [(idx, dist) for (idx, species, dist) in neighbors if species == sp]
        cand.sort(key=lambda x: x[1])
        if len(cand) < req:
            return None
        chosen_neighbors.extend(idx for idx, _ in cand[:req])

    return {
        "motif_name": pattern.name,
        "central_index": central_index,
        "neighbor_indices": sorted(set(chosen_neighbors)),
    }


def find_motif_occurrences(
    structure: Structure,
    pattern: MotifPattern,
) -> List[Dict]:
    """
    Find all occurrences of `pattern` in `structure`.
    """
    matches: List[Dict] = []
    for i in range(len(structure)):
        m = _match_pattern(structure, i, pattern)
        if m is not None:
            matches.append(m)
    return matches


def decompose_structure(
    structure: Structure,
    motif_library: Sequence[MotifPattern],
    allowed_motifs: Optional[Sequence[str]] = None,
    allow_overlap: bool = True,
) -> Dict:
    """
    Decompose a structure into motif instances from `motif_library`.
    """
    if allowed_motifs is not None:
        allowed = set(allowed_motifs)
        motifs = [m for m in motif_library if m.name in allowed]
    else:
        motifs = list(motif_library)

    raw_matches: List[Dict] = []
    for pattern in motifs:
        occs = find_motif_occurrences(structure, pattern)
        raw_matches.extend(occs)

    if allow_overlap:
        assigned = raw_matches
    else:
        # Greedy: prefer motifs with largest number of sites
        raw_matches.sort(key=lambda m: len(m["neighbor_indices"]) + 1, reverse=True)
        used_sites = set()
        assigned = []
        for m in raw_matches:
            sites = {m["central_index"], *m["neighbor_indices"]}
            if sites & used_sites:
                continue
            assigned.append(m)
            used_sites |= sites

    all_sites = set(range(len(structure)))
    covered_sites = set()
    for m in assigned:
        covered_sites.add(m["central_index"])
        covered_sites.update(m["neighbor_indices"])

    unassigned_sites = sorted(all_sites - covered_sites)

    return {
        "motifs": assigned,
        "unassigned_sites": unassigned_sites,
    }


# =========================
# Structure comparison logic
# =========================

def compare_two_structures(
    structure1: Structure,
    structure2: Structure,
    motif_library: Sequence[MotifPattern],
    allowed_motifs: Optional[Sequence[str]] = None,
    allow_overlap: bool = True,
) -> Dict:
    """
    Decompose both structures and compare motifs.
    """
    decomp1 = decompose_structure(
        structure1, motif_library, allowed_motifs=allowed_motifs, allow_overlap=allow_overlap
    )
    decomp2 = decompose_structure(
        structure2, motif_library, allowed_motifs=allowed_motifs, allow_overlap=allow_overlap
    )

    index1: Dict[str, List[Dict]] = defaultdict(list)
    index2: Dict[str, List[Dict]] = defaultdict(list)

    for m in decomp1["motifs"]:
        index1[m["motif_name"]].append(m)
    for m in decomp2["motifs"]:
        index2[m["motif_name"]].append(m)

    names1 = set(index1.keys())
    names2 = set(index2.keys())

    shared = []
    for name in sorted(names1 & names2):
        shared.append(
            {
                "motif_name": name,
                "count_1": len(index1[name]),
                "count_2": len(index2[name]),
                "instances_1": index1[name],
                "instances_2": index2[name],
            }
        )

    result = {
        "decomp_1": decomp1,
        "decomp_2": decomp2,
        "shared_motifs": shared,
        "unique_to_1": sorted(names1 - names2),
        "unique_to_2": sorted(names2 - names1),
    }
    return result


# =========================
# Small summaries for the LLM
# =========================

def _summarise_decomposition_for_llm(result: Dict, max_examples_per_motif: int = 3) -> Dict:
    """
    Compress a full decomposition result into something LLM-friendly:
    - counts per motif_name
    - total motif instances
    - number of unassigned sites
    - a few example instances per motif
    """
    motifs: List[Dict] = result.get("motifs", [])
    unassigned_sites = result.get("unassigned_sites", [])

    counts: Dict[str, int] = defaultdict(int)
    examples: Dict[str, List[Dict]] = defaultdict(list)
    for m in motifs:
        name = m.get("motif_name", "UNKNOWN")
        counts[name] += 1
        if len(examples[name]) < max_examples_per_motif:
            examples[name].append(
                {
                    "central_index": int(m.get("central_index", -1)),
                    "neighbor_indices": m.get("neighbor_indices", []),
                }
            )

    return {
        "total_motif_instances": len(motifs),
        "motif_counts": dict(counts),
        "unassigned_sites_count": len(unassigned_sites),
        "unassigned_sites_sample": unassigned_sites[:20],
        "example_instances": {k: v for k, v in examples.items()},
    }


def _summarise_comparison_for_llm(result: Dict, max_examples_per_motif: int = 3) -> Dict:
    """
    Compress the full comparison result:
    - summaries of decomp_1 and decomp_2
    - shared motif names + counts
    - unique motif names
    """
    decomp1 = result.get("decomp_1", {})
    decomp2 = result.get("decomp_2", {})
    shared = result.get("shared_motifs", [])
    unique1 = result.get("unique_to_1", [])
    unique2 = result.get("unique_to_2", [])

    summary_decomp1 = _summarise_decomposition_for_llm(
        {"motifs": decomp1.get("motifs", []), "unassigned_sites": decomp1.get("unassigned_sites", [])},
        max_examples_per_motif=max_examples_per_motif,
    )
    summary_decomp2 = _summarise_decomposition_for_llm(
        {"motifs": decomp2.get("motifs", []), "unassigned_sites": decomp2.get("unassigned_sites", [])},
        max_examples_per_motif=max_examples_per_motif,
    )

    shared_summary = []
    for s in shared:
        shared_summary.append(
            {
                "motif_name": s.get("motif_name", "UNKNOWN"),
                "count_1": int(s.get("count_1", 0)),
                "count_2": int(s.get("count_2", 0)),
            }
        )

    return {
        "decomp_1_summary": summary_decomp1,
        "decomp_2_summary": summary_decomp2,
        "shared_motifs": shared_summary,
        "unique_to_1": unique1,
        "unique_to_2": unique2,
    }


# =========================
# ChemCrow tools
# =========================

class MotifDecompositionTool(BaseTool):
    """
    ChemCrow tool for motif analysis in CIF files.

    INPUT FORMAT (query string):
    -----------------------------
    JSON with keys:

      Required:
        - "mode": one of ["search", "all", "from-list"]
        - "cif_path": path to the CIF file on disk

      Motif definitions: provide EITHER:
        - "motifs": list of motif definitions, OR
        - "motif_library_path": path to JSON motifs file

      Optional:
        - "motif_name" (for mode == "search")
        - "allowed_motifs" (for mode == "from-list")
        - "allow_overlap": bool

    OUTPUT:
    -------
    A small JSON summary **plus** a path to the full JSON file on disk
    with all motif instances and unassigned sites.
    """

    name: str = "MotifDecomposition"
    description: str = (
        "Analyze coordination motifs in a CIF file. "
        "Input MUST be a JSON string with keys: "
        "'mode' (search|all|from-list), 'cif_path', and "
        "either 'motifs' or 'motif_library_path'. "
        "Returns a compact summary and also saves the **full** result "
        "to a JSON file on disk for offline inspection."
    )

    default_motif_library_path: Optional[str] = None

    def __init__(self, default_motif_library_path: Optional[str] = None):
        super().__init__()
        self.default_motif_library_path = default_motif_library_path

    def _run(self, query: str) -> str:
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return (
                "Invalid input for MotifDecomposition. "
                "Expected a JSON string. Example:\n"
                '{ "mode": "all", "cif_path": "path/to/file.cif", '
                '"motif_library_path": "path/to/motifs.json" }'
            )

        mode = params.get("mode", "all")
        cif_path = params.get("cif_path", None)
        if cif_path is None:
            return "Error: 'cif_path' is required."

        motif_defs = params.get("motifs", None)
        motif_library_path = params.get("motif_library_path", self.default_motif_library_path)
        allow_overlap = params.get("allow_overlap", True)

        try:
            motif_library = _load_motif_library(
                motifs=motif_defs,
                motif_library_path=motif_library_path,
            )
        except Exception as e:
            return f"Error loading motif library: {e}"

        try:
            structure = Structure.from_file(cif_path)
        except Exception as e:
            return f"Error reading CIF file '{cif_path}': {e}"

        # Build full internal result
        if mode == "search":
            motif_name = params.get("motif_name", None)
            if not motif_name:
                return "Error: 'motif_name' is required in 'search' mode."
            try:
                pattern = next(m for m in motif_library if m.name == motif_name)
            except StopIteration:
                return f"Error: motif '{motif_name}' not found in library."
            occs = find_motif_occurrences(structure, pattern)
            full_result = {
                "mode": "search",
                "motif_name": motif_name,
                "motifs": occs,
                "unassigned_sites": [],
            }

        elif mode == "from-list":
            allowed = params.get("allowed_motifs", None)
            if not allowed:
                return "Error: 'allowed_motifs' is required in 'from-list' mode."
            decomp = decompose_structure(
                structure,
                motif_library,
                allowed_motifs=allowed,
                allow_overlap=allow_overlap,
            )
            decomp["mode"] = "from-list"
            full_result = decomp

        elif mode == "all":
            decomp = decompose_structure(
                structure,
                motif_library,
                allowed_motifs=None,
                allow_overlap=allow_overlap,
            )
            decomp["mode"] = "all"
            full_result = decomp

        else:
            return "Error: 'mode' must be one of 'search', 'all', or 'from-list'."

        # Save full result to disk
        out_dir = Path("motif_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(cif_path).stem
        out_path = out_dir / f"{base}_motifs_{mode}.json"
        out_path.write_text(json.dumps(full_result, indent=2, default=_json_default), encoding="utf-8")

        # Build small summary for the LLM
        summary_core = _summarise_decomposition_for_llm(full_result)
        summary = {
            "mode": mode,
            "cif_path": cif_path,
            "full_result_path": str(out_path),
            **summary_core,
        }

        return json.dumps(summary, indent=2, default=_json_default)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("this tool does not support async")


class MotifComparisonTool(BaseTool):
    """
    Compare motifs between TWO CIF files using a motif library.

    INPUT (query string):
    ---------------------
    JSON with:

      Required:
        - "cif_path_1": path to first CIF
        - "cif_path_2": path to second CIF

      Motif definitions: provide EITHER:
        - "motifs": list of motif definitions, OR
        - "motif_library_path": path to JSON file of motifs

      Optional:
        - "allowed_motifs": list of motif names to consider (subset)
        - "allow_overlap": bool, default True

    OUTPUT:
    -------
    A small JSON summary **plus** a path to a full JSON file on disk.
    """

    name: str = "MotifComparison"
    description: str = (
        "Compare motifs between two CIF files using a motif library. "
        "Input MUST be a JSON string with keys: "
        "'cif_path_1', 'cif_path_2', and either 'motifs' or 'motif_library_path'. "
        "Returns a compact summary and saves the full comparison result to disk."
    )

    default_motif_library_path: Optional[str] = None

    def __init__(self, default_motif_library_path: Optional[str] = None):
        super().__init__()
        self.default_motif_library_path = default_motif_library_path

    def _run(self, query: str) -> str:
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return (
                "Invalid input for MotifComparison. Expected a JSON string. Example:\n"
                '{ "cif_path_1": "path/to/a.cif", '
                '"cif_path_2": "path/to/b.cif", '
                '"motif_library_path": "path/to/motifs.json" }'
            )

        cif_path_1 = params.get("cif_path_1", None)
        cif_path_2 = params.get("cif_path_2", None)
        if not cif_path_1 or not cif_path_2:
            return "Error: 'cif_path_1' and 'cif_path_2' are both required."

        motif_defs = params.get("motifs", None)
        motif_library_path = params.get("motif_library_path", self.default_motif_library_path)
        allowed_motifs = params.get("allowed_motifs", None)
        allow_overlap = params.get("allow_overlap", True)

        try:
            motif_library = _load_motif_library(
                motifs=motif_defs,
                motif_library_path=motif_library_path,
            )
        except Exception as e:
            return f"Error loading motif library: {e}"

        try:
            struct1 = Structure.from_file(cif_path_1)
        except Exception as e:
            return f"Error reading CIF file 1 '{cif_path_1}': {e}"

        try:
            struct2 = Structure.from_file(cif_path_2)
        except Exception as e:
            return f"Error reading CIF file 2 '{cif_path_2}': {e}"

        try:
            full_result = compare_two_structures(
                struct1,
                struct2,
                motif_library,
                allowed_motifs=allowed_motifs,
                allow_overlap=allow_overlap,
            )
        except Exception as e:
            return f"Error comparing motifs: {e}"

        # Save full comparison result to disk
        out_dir = Path("motif_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        base1 = Path(cif_path_1).stem
        base2 = Path(cif_path_2).stem
        out_path = out_dir / f"compare_{base1}_vs_{base2}.json"
        out_path.write_text(json.dumps(full_result, indent=2, default=_json_default), encoding="utf-8")

        # Small summary for LLM
        summary_core = _summarise_comparison_for_llm(full_result)
        summary = {
            "cif_path_1": cif_path_1,
            "cif_path_2": cif_path_2,
            "full_result_path": str(out_path),
            **summary_core,
        }

        return json.dumps(summary, indent=2, default=_json_default)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("this tool does not support async")
