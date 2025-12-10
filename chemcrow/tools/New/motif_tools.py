import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from langchain.tools import BaseTool
from pymatgen.core import Structure
from pymatgen.analysis.local_env import MinimumDistanceNN


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
    extra: Dict[str, str] = field(default_factory=dict)


def _load_motif_library_from_file(path: str | Path) -> List[MotifPattern]:
    data = json.loads(Path(path).read_text())
    return [MotifPattern(**entry) for entry in data]


def _load_motif_library(
    motifs: Optional[Sequence[Dict]] = None,
    motif_library_path: Optional[str] = None,
) -> List[MotifPattern]:
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
    # get_nn_info returns list of dicts, each with site_index, site, weight, etc.
    neighs = nn_strategy.get_nn_info(structure, central_index)

    neighbors = []
    central_site = structure[central_index]
    for info in neighs:
        j = info["site_index"]
        dist = central_site.distance(structure[j])
        if dist <= max_distance:
            species_j = structure[j].specie.symbol if hasattr(structure[j], "specie") else structure[j].species_string
            neighbors.append((j, species_j, dist))

    return neighbors


def _match_pattern(
    structure: Structure,
    central_index: int,
    pattern: MotifPattern,
) -> Optional[Dict]:
    # Check central species first
    central_site = structure[central_index]
    central_species = central_site.specie.symbol if hasattr(central_site, "specie") else central_site.species_string
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

    Parameters
    ----------
    allowed_motifs:
        If provided, only motifs whose name is in this list are considered.
    allow_overlap:
        If False, each site can only appear in at most one motif (greedy).
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
# ChemCrow tool wrapper
# =========================

class MotifDecompositionTool(BaseTool):
    """
    ChemCrow tool for motif analysis in CIF files.

    INPUT FORMAT (query string):
    -----------------------------
    The input MUST be a JSON string with the following keys:

    Required keys:
      - "mode": one of ["search", "all", "from-list"]
      - "cif_path": path to the CIF file on disk

    Motif definitions: provide EITHER:
      - "motifs": a list of JSON motifs of the form:
            {
              "name": "TiO6_oct",
              "central_species": "Ti",
              "neighbor_species_counts": {"O": 6},
              "max_distance": 2.3
            }
        OR
      - "motif_library_path": path to a JSON file containing such entries.

    Additional keys depending on mode:
      - mode == "search":
          "motif_name": name of motif to search for
      - mode == "from-list":
          "allowed_motifs": list of motif names to use

    Optional:
      - "allow_overlap": bool, default True (if False, each site is used in at most one motif)

    OUTPUT:
    -------
    A JSON string with motif instances and (if decomposition) unassigned sites. Examples:

    - search:
      {
        "mode": "search",
        "motif_name": "TiO6_oct",
        "occurrences": [
          {
            "motif_name": "TiO6_oct",
            "central_index": 10,
            "neighbor_indices": [3, 5, 7, 12, 14, 18]
          },
          ...
        ]
      }

    - all/from-list:
      {
        "mode": "all",
        "motifs": [...],
        "unassigned_sites": [...]
      }
    """

    name: str = "MotifDecomposition"
    description: str = (
        "Analyze coordination motifs in a CIF file. "
        "Input MUST be a JSON string with keys: "
        "'mode' (search|all|from-list), 'cif_path', and "
        "either 'motifs' (inline motif definitions) or "
        "'motif_library_path' (JSON file). "
        "For 'search', also provide 'motif_name'. "
        "For 'from-list', provide 'allowed_motifs'. "
        "Returns motif instances and unassigned site indices as JSON."
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

        if mode == "search":
            motif_name = params.get("motif_name", None)
            if not motif_name:
                return "Error: 'motif_name' is required in 'search' mode."
            try:
                pattern = next(m for m in motif_library if m.name == motif_name)
            except StopIteration:
                return f"Error: motif '{motif_name}' not found in library."
            occs = find_motif_occurrences(structure, pattern)
            result = {
                "mode": "search",
                "motif_name": motif_name,
                "occurrences": occs,
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
            result = {
                "mode": "from-list",
                "allowed_motifs": allowed,
                **decomp,
            }

        elif mode == "all":
            decomp = decompose_structure(
                structure,
                motif_library,
                allowed_motifs=None,
                allow_overlap=allow_overlap,
            )
            result = {
                "mode": "all",
                **decomp,
            }

        else:
            return "Error: 'mode' must be one of 'search', 'all', or 'from-list'."

        return json.dumps(result, indent=2)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
