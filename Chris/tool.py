import os
import textwrap
import traceback

from chemcrow.agents import make_tools
from langchain.chat_models import ChatOpenAI


# ---------- LLM FOR TOOLS ----------

def make_tools_llm():
    """LLM used by ChemCrow tools (Wikipedia, LiteratureSearch, etc.)."""
    return ChatOpenAI(
        model_name="gpt-4.1-mini",  # or gpt-4o-mini, etc.
        temperature=0.0,
        request_timeout=1000,
    )


# ---------- BUILD TOOLS ----------

def build_tools():
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "SEMANTIC_SCHOLAR_API_KEY": os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        "SERPAPI_API_KEY": os.environ.get("SERPAPI_API_KEY", ""),
        "RXN4CHEM_API_KEY": os.environ.get("RXN4CHEM_API_KEY", ""),
        "POSTERA_API_KEY": os.environ.get("POSTERA_API_KEY", ""),
        "CHEMSPACE_API_KEY": os.environ.get("CHEMSPACE_API_KEY", ""),
    }

    tools_llm = make_tools_llm()

    tools = make_tools(
        tools_llm,
        api_keys=api_keys,
        verbose=False,
    )
    return tools, api_keys


# ---------- TEST INPUTS PER TOOL ----------

def get_test_input(tool_name: str) -> str | None:
    """Return a tool-specific test input, or None if we don't have one."""
    # Some valid SMILES we can reuse
    PARACETAMOL_SMILES = "CC(=O)NC1=CC=C(C=C1)O"
    ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
    ETHANOL_SMILES = "CCO"

    tests = {
        # General
        "Python_REPL": "2 + 2",
        "Wikipedia": "Paracetamol",

        # Converters / identifiers
        "Name2SMILES": "aspirin",
        "Mol2CAS": "paracetamol",
        "SMILES2Name": ASPIRIN_SMILES,
        "SMILES2Weight": PARACETAMOL_SMILES,

        # Functional / safety / similarity
        "FunctionalGroups": ETHANOL_SMILES,
        "ExplosiveCheck": "C(C(=O)O[N+](=O)[O-])N",  # nitro-ish
        "ControlChemCheck": ASPIRIN_SMILES,
        "SimilarityToControlChem": PARACETAMOL_SMILES,
        "SafetySummary": PARACETAMOL_SMILES,
        # expects "SMI1.SMI2"
        "MolSimilarity": f"{PARACETAMOL_SMILES}.{ASPIRIN_SMILES}",

        # IP / literature
        "PatentCheck": "paracetamol",
        "LiteratureSearch": "melting point of paracetamol",
    }

    return tests.get(tool_name)


# ---------- HEURISTICS FOR PASS / FAIL / SKIP ----------

ERROR_SUBSTRINGS = [
    "invalid smiles",
    "wrong argument",
    "please input a valid",
    "molecule not found",
    "could not find a molecule matching the text",
    "error:",
]

def looks_like_error(output: str) -> bool:
    text = output.lower()
    return any(s in text for s in ERROR_SUBSTRINGS)


def tool_needs_missing_key(tool, api_keys: dict) -> bool:
    """
    Crude heuristic: if description mentions an external API and we don't have its key,
    treat this as SKIP (not a tool bug, just missing credentials).
    """
    desc = (getattr(tool, "description", "") or "").lower()

    patterns = {
        "semantic scholar": "SEMANTIC_SCHOLAR_API_KEY",
        "serpapi": "SERPAPI_API_KEY",
        "rxn for chemistry": "RXN4CHEM_API_KEY",
        "rxn ": "RXN4CHEM_API_KEY",
        "postera": "POSTERA_API_KEY",
        "chemscape": "CHEMSPACE_API_KEY",
        "chemspec": "CHEMSPACE_API_KEY",
        "chems space": "CHEMSPACE_API_KEY",
    }

    for substr, key in patterns.items():
        if substr in desc and not api_keys.get(key):
            return True

    return False


def summarize_output(text: str, max_len: int = 500) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


# ---------- MAIN ----------

def main():
    print("ChemCrow tool smoke tests\n")

    tools, api_keys = build_tools()

    print("Loaded tools:")
    for t in tools:
        print(f"  - {t.name}")
    print("\n================ RUNNING TESTS ================\n")

    results = {}

    for tool in tools:
        name = tool.name
        test_input = get_test_input(name)

        print(f"\n--- Tool: {name} ---")

        if tool_needs_missing_key(tool, api_keys):
            print("  [SKIP] Missing required API key for this tool.")
            results[name] = {
                "status": "SKIP",
                "reason": "missing_api_key",
                "output": None,
            }
            continue

        if test_input is None:
            print("  [SKIP] No test case defined yet for this tool.")
            results[name] = {
                "status": "SKIP",
                "reason": "no_test_case",
                "output": None,
            }
            continue

        print(f"  Test input: {test_input!r}")

        try:
            raw_output = tool.run(test_input)
            out_str = str(raw_output)
            errorish = looks_like_error(out_str)
            status = "PASS" if not errorish else "FAIL_OUTPUT"
        except Exception as e:
            raw_output = f"[EXCEPTION] {type(e).__name__}: {e}"
            out_str = raw_output
            status = "FAIL_EXCEPTION"

        print(f"  Status: {status}")
        print("  Output:")
        print(
            textwrap.indent(
                summarize_output(out_str),
                prefix="    ",
            )
        )

        results[name] = {
            "status": status,
            "output": out_str,
        }

    print("\n================ SUMMARY ================\n")
    for name, info in results.items():
        print(f"{name}: {info['status']}")

    print("\n================ SUGGESTED WHITELIST ================\n")
    good_tools = [name for name, info in results.items() if info["status"] == "PASS"]
    print("Tools that look OK to keep in a 'clean' ChemCrow instance:")
    for name in good_tools:
        print("  -", name)

    print("\nYou can use this snippet in your own ChemCrow setup:")
    print("    from chemcrow.agents import make_tools, ChemCrow")
    print("    tools = make_tools(tools_llm, api_keys=api_keys, local_rxn=False)")
    print("    allowed = {")
    for name in good_tools:
        print(f"        {name!r},")
    print("    }")
    print("    tools = [t for t in tools if t.name in allowed]")
    print("    chem_model = ChemCrow(tools=tools, ...)")


if __name__ == "__main__":
    main()
