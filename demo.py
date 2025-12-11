"""
Demo script for Crystal Materials Tools with LLM Integration

This script demonstrates all crystal materials tools available in the ChemCrow AI agent.
Uses an actual CIF file and shows both basic and LLM-integrated tools in a workflow.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from langchain.chat_models import ChatOpenAI

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Note: LangChain not available. LLM-integrated tools will be skipped.")

# Import all crystal materials tools
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
)

# Import LLM-integrated tools
try:
    from chemcrow.tools.crystal_materials_llm import (
        LLMStructureFromCIF,
        LLMStructureSymmetryAnalysis,
        LLMStructureProperties,
        LLMCompositionAnalysis,
        LLMStructureComparison,
        LLMStructureSubstitution,
        LLMSurfaceGeneration,
        LLMCoordinationAnalysis,
        LLMLatticeParameterOptimization,
    )

    LLM_TOOLS_AVAILABLE = True
except ImportError:
    LLM_TOOLS_AVAILABLE = False


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def demo_basic_tools(cif_path):
    """Demonstrate basic crystal materials tools (without LLM)."""
    print_separator("BASIC TOOLS DEMONSTRATION")
    print(f"Using CIF file: {cif_path}")

    if not Path(cif_path).exists():
        print(f"ERROR: CIF file not found: {cif_path}")
        return

    tools_demo = {
        "StructureFromCIF": (StructureFromCIF(), cif_path),
        "StructureSymmetryAnalysis": (StructureSymmetryAnalysis(), cif_path),
        "StructureProperties": (StructureProperties(), cif_path),
        "CompositionAnalysis": (CompositionAnalysis(), "C12H6N4"),  # Example from CIF
        "PrimitiveCellConversion": (PrimitiveCellConversion(), cif_path),
        "CoordinationAnalysis": (CoordinationAnalysis(), cif_path),
        "LatticeParameterOptimization": (LatticeParameterOptimization(), cif_path),
    }

    results = {}
    for tool_name, (tool, input_data) in tools_demo.items():
        print(f"\n--- {tool_name} ---")
        print(f"Input: {input_data}")
        print("-" * 80)
        try:
            result = tool._run(input_data)
            # Truncate long outputs for display
            if len(result) > 500:
                print(result[:500] + "\n... (truncated)")
            else:
                print(result)
            results[tool_name] = "SUCCESS"
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results[tool_name] = f"ERROR: {str(e)}"
        print()

    # Demo tools that need multiple inputs
    print("\n--- StructureComparison ---")
    print(f"Input: {cif_path}|{cif_path} (comparing structure to itself)")
    print("-" * 80)
    try:
        comp_tool = StructureComparison()
        result = comp_tool._run(f"{cif_path}|{cif_path}")
        print(result[:500] if len(result) > 500 else result)
        results["StructureComparison"] = "SUCCESS"
    except Exception as e:
        print(f"ERROR: {str(e)}")
        results["StructureComparison"] = f"ERROR: {str(e)}"

    print("\n--- StructureSubstitution ---")
    print(f"Input: {cif_path}|C|Si")
    print("-" * 80)
    try:
        sub_tool = StructureSubstitution()
        result = sub_tool._run(f"{cif_path}|C|Si")
        print(result[:500] if len(result) > 500 else result)
        results["StructureSubstitution"] = "SUCCESS"
    except Exception as e:
        print(f"ERROR: {str(e)}")
        results["StructureSubstitution"] = f"ERROR: {str(e)}"

    print("\n--- SurfaceGeneration ---")
    print(f"Input: {cif_path}|1 0 0|10|15")
    print("-" * 80)
    try:
        surf_tool = SurfaceGeneration()
        result = surf_tool._run(f"{cif_path}|1 0 0|10|15")
        print(result[:500] if len(result) > 500 else result)
        results["SurfaceGeneration"] = "SUCCESS"
    except Exception as e:
        print(f"ERROR: {str(e)}")
        results["SurfaceGeneration"] = f"ERROR: {str(e)}"

    # Export to CIF
    print("\n--- StructureToCIF ---")
    output_cif = str(Path(cif_path).parent / "exported_output.cif")
    print(f"Input: {cif_path}|{output_cif}")
    print("-" * 80)
    try:
        export_tool = StructureToCIF()
        result = export_tool._run(f"{cif_path}|{output_cif}")
        print(result)
        if Path(output_cif).exists():
            print(f"✓ File created: {output_cif}")
        results["StructureToCIF"] = "SUCCESS"
    except Exception as e:
        print(f"ERROR: {str(e)}")
        results["StructureToCIF"] = f"ERROR: {str(e)}"

    return results


def demo_llm_integrated_tools(cif_path, llm):
    """Demonstrate LLM-integrated crystal materials tools as Q&A pairs."""
    print_separator("LLM-INTEGRATED TOOLS DEMONSTRATION (Q&A FORMAT)")
    print(f"Using CIF file: {cif_path}")
    print("These tools use LLM to summarize and reason about the results.")
    print("Format: QUESTION (prompt to LLM) → ANSWER (LLM interpretation)")

    if not Path(cif_path).exists():
        print(f"ERROR: CIF file not found: {cif_path}")
        return

    if not LLM_TOOLS_AVAILABLE:
        print("ERROR: LLM-integrated tools not available.")
        return

    # Define questions/prompts for each tool
    tools_qa = [
        {
            "tool_name": "LLMStructureFromCIF",
            "question": f"What are the key structural features, properties, and characteristics of the crystal structure in {cif_path}?",
            "tool": LLMStructureFromCIF(llm),
            "input": cif_path,
        },
        {
            "tool_name": "LLMStructureSymmetryAnalysis",
            "question": f"Analyze and explain the symmetry properties (space group, crystal system, point group) of the structure in {cif_path}.",
            "tool": LLMStructureSymmetryAnalysis(llm),
            "input": cif_path,
        },
        {
            "tool_name": "LLMStructureProperties",
            "question": f"Interpret the physical and chemical properties (density, volume, coordination) of the structure in {cif_path}. What do these properties indicate about the material?",
            "tool": LLMStructureProperties(llm),
            "input": cif_path,
        },
        {
            "tool_name": "LLMCompositionAnalysis",
            "question": "Explain the chemical composition C12H6N4. What are the element ratios, oxidation states, and chemical characteristics?",
            "tool": LLMCompositionAnalysis(llm),
            "input": "C12H6N4",
        },
        {
            "tool_name": "LLMCoordinationAnalysis",
            "question": f"Analyze the coordination environments in {cif_path}. What are the coordination numbers, bonding patterns, and structural insights?",
            "tool": LLMCoordinationAnalysis(llm),
            "input": cif_path,
        },
        {
            "tool_name": "LLMLatticeParameterOptimization",
            "question": f"Analyze the lattice parameters of {cif_path}. What are the lattice type, volume considerations, and optimization suggestions?",
            "tool": LLMLatticeParameterOptimization(llm),
            "input": cif_path,
        },
        {
            "tool_name": "LLMStructureComparison",
            "question": f"Compare the structure in {cif_path} with itself. Are they similar? What are the key observations?",
            "tool": LLMStructureComparison(llm),
            "input": f"{cif_path}|{cif_path}",
        },
        {
            "tool_name": "LLMStructureSubstitution",
            "question": f"If we substitute Carbon with Silicon in {cif_path}, what changes occur in structure and properties? What are the implications?",
            "tool": LLMStructureSubstitution(llm),
            "input": f"{cif_path}|C|Si",
        },
        {
            "tool_name": "LLMSurfaceGeneration",
            "question": f"Generate a (100) surface from {cif_path}. What are the surface characteristics and potential applications for surface science studies?",
            "tool": LLMSurfaceGeneration(llm),
            "input": f"{cif_path}|1 0 0|10|15",
        },
    ]

    results = {}
    for idx, qa_pair in enumerate(tools_qa, 1):
        tool_name = qa_pair["tool_name"]
        question = qa_pair["question"]
        tool = qa_pair["tool"]
        input_data = qa_pair["input"]

        print(f"\n{'='*80}")
        print(f"Q&A {idx}: {tool_name}")
        print("=" * 80)
        print(f"\n📝 QUESTION/PROMPT:")
        print(f"   {question}")
        print(f"\n🔧 Tool Input: {input_data}")
        print("\n" + "-" * 80)
        print("\n💡 ANSWER (LLM Interpretation):\n")

        try:
            result = tool._run(input_data)
            # Extract LLM answer from the result
            if "Raw Data:" in result and (
                "LLM Summary:" in result
                or "LLM Analysis:" in result
                or "LLM Interpretation:" in result
                or "LLM Explanation:" in result
            ):
                # Split on any of the LLM markers
                for marker in [
                    "LLM Summary:",
                    "LLM Analysis:",
                    "LLM Interpretation:",
                    "LLM Explanation:",
                ]:
                    if marker in result:
                        parts = result.split(f"\n\n{marker}")
                        if len(parts) > 1:
                            llm_answer = parts[1].strip()
                            print(llm_answer)
                        else:
                            print(result)
                        break
                else:
                    print(result)
            else:
                print(result)

            results[tool_name] = "SUCCESS"
            print(f"\n✓ Successfully analyzed using {tool_name}")
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results[tool_name] = f"ERROR: {str(e)}"
        print()

    return results


def run_workflow_demo():
    """Run a complete workflow demonstration using all tools."""
    print_separator("CRYSTAL MATERIALS TOOLS - COMPLETE WORKFLOW DEMONSTRATION")

    load_dotenv()

    # Use the actual CIF file
    cif_path = Path("tests/tmp_folder/13030N2_ddec.cif")

    if not cif_path.exists():
        print(f"ERROR: CIF file not found at {cif_path}")
        print("Please ensure the file exists at the specified path.")
        return

    print(f"Test CIF file: {cif_path.absolute()}")
    print(f"File exists: {cif_path.exists()}")
    print(f"File size: {cif_path.stat().st_size} bytes\n")

    # Results tracking
    all_results = {}

    # Step 1: Basic tools
    print_separator("STEP 1: BASIC TOOLS")
    basic_results = demo_basic_tools(str(cif_path))
    all_results.update(basic_results)

    # Step 2: LLM-integrated tools (if available)
    if LLM_AVAILABLE and LLM_TOOLS_AVAILABLE:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            print_separator("STEP 2: LLM-INTEGRATED TOOLS")
            llm = ChatOpenAI(
                temperature=0.0,
                model_name="gpt-3.5-turbo",
                openai_api_key=openai_api_key,
            )
            llm_results = demo_llm_integrated_tools(str(cif_path), llm)
            all_results.update(llm_results)
        else:
            print_separator("STEP 2: LLM-INTEGRATED TOOLS")
            print("⚠ SKIPPED: OPENAI_API_KEY not found in environment variables")
            print("Set OPENAI_API_KEY to use LLM-integrated tools")
    else:
        print_separator("STEP 2: LLM-INTEGRATED TOOLS")
        print("⚠ SKIPPED: LLM tools not available")
        if not LLM_AVAILABLE:
            print("Install langchain: pip install langchain")
        if not LLM_TOOLS_AVAILABLE:
            print("LLM-integrated crystal tools module not available")

    # Summary
    print_separator("DEMONSTRATION SUMMARY")

    print("Basic Tools Results:")
    for tool_name, status in basic_results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"  {status_symbol} {tool_name}: {status}")

    if LLM_AVAILABLE and LLM_TOOLS_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        print("\nLLM-Integrated Tools Results:")
        llm_tools = {k: v for k, v in all_results.items() if k.startswith("LLM")}
        for tool_name, status in llm_tools.items():
            status_symbol = "✓" if status == "SUCCESS" else "✗"
            print(f"  {status_symbol} {tool_name}: {status}")

    print(f"\nTotal tools tested: {len(all_results)}")
    successful = sum(1 for v in all_results.values() if v == "SUCCESS")
    print(f"Successful: {successful}")
    print(f"Failed: {len(all_results) - successful}")

    print("\n" + "=" * 80)
    print("Workflow demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_workflow_demo()
