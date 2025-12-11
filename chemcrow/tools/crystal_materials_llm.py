"""
LLM-integrated crystal materials tools with summarization and reasoning.

These tools wrap the base crystal materials tools and use LLMChain to provide
natural language summaries and reasoning about the results.
"""

import json
from typing import Optional

from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool

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


# Prompt templates for LLM summarization
CRYSTAL_STRUCTURE_SUMMARY_PROMPT = """
You are an expert in crystallography and materials science. 
Analyze the following crystal structure data and provide a clear, concise summary.

Crystal Structure Data:
{data}

Provide a summary that includes:
1. Key structural features (composition, crystal system, space group)
2. Important properties (density, volume, coordination)
3. Notable characteristics or insights
4. Any recommendations or observations

Be concise but informative. Use scientific terminology appropriately.
Summary:"""

SYMMETRY_ANALYSIS_PROMPT = """
You are an expert crystallographer. Analyze the following symmetry analysis results.

Symmetry Analysis Data:
{data}

Provide a clear summary explaining:
1. The space group and what it means
2. The crystal system
3. The significance of the symmetry operations
4. Whether the structure is primitive or not

Summary:"""

PROPERTIES_ANALYSIS_PROMPT = """
You are a materials scientist. Analyze the following material properties.

Material Properties Data:
{data}

Provide an interpretation that includes:
1. Key physical properties and their values
2. What these properties indicate about the material
3. Comparison to typical values (if relevant)
4. Potential applications or implications

Summary:"""

COMPOSITION_ANALYSIS_PROMPT = """
You are a chemist. Analyze the following composition data.

Composition Data:
{data}

Provide an explanation that includes:
1. The chemical formula and its meaning
2. Element ratios and their significance
3. Oxidation states (if available)
4. Chemical characteristics based on composition

Summary:"""

COMPARISON_ANALYSIS_PROMPT = """
You are a materials scientist. Compare the following two crystal structures.

Comparison Data:
{data}

Provide a comparison that includes:
1. Whether the structures are similar or different
2. Key differences in composition and structure
3. Implications of the comparison
4. Any notable observations

Summary:"""

SUBSTITUTION_ANALYSIS_PROMPT = """
You are a materials scientist. Analyze the following element substitution.

Substitution Data:
{data}

Provide an analysis that includes:
1. The substitution performed (old element → new element)
2. Changes in structure and properties
3. Implications of the substitution
4. Potential effects on material behavior

Summary:"""

SURFACE_ANALYSIS_PROMPT = """
You are a surface scientist. Analyze the following surface generation results.

Surface Data:
{data}

Provide an analysis that includes:
1. The surface orientation (Miller indices)
2. Surface characteristics (area, thickness)
3. Potential applications or studies this surface could be used for
4. Important considerations for surface science calculations

Summary:"""

COORDINATION_ANALYSIS_PROMPT = """
You are a crystallographer. Analyze the following coordination environment data.

Coordination Data:
{data}

Provide an analysis that includes:
1. The coordination numbers and their distribution
2. Local environment characteristics
3. Bonding patterns
4. Structural insights based on coordination

Summary:"""

LATTICE_ANALYSIS_PROMPT = """
You are a materials scientist. Analyze the following lattice parameters.

Lattice Parameter Data:
{data}

Provide an analysis that includes:
1. The lattice type and parameters
2. Volume and density considerations
3. Suggestions for optimization
4. Structural insights

Summary:"""


class LLMStructureFromCIF(BaseTool):
    """Load crystal structure from CIF with LLM summary."""

    name = "LLMStructureFromCIF"
    description = (
        "Load a crystal structure from a CIF file and provide a natural language summary. "
        "Input: path to CIF file. "
        "Returns: structured data and LLM-generated summary."
    )

    llm: BaseLanguageModel = None
    llm_chain: LLMChain = None
    base_tool: StructureFromCIF = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.base_tool = StructureFromCIF()
        self.llm = llm
        prompt = PromptTemplate(
            template=CRYSTAL_STRUCTURE_SUMMARY_PROMPT, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, cif_path: str) -> str:
        """Load structure and get LLM summary."""
        try:
            raw_result = self.base_tool._run(cif_path)
            summary = self.llm_chain.run(data=raw_result)
            return f"Raw Data:\n{raw_result}\n\nLLM Summary:\n{summary}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, cif_path: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class LLMStructureSymmetryAnalysis(BaseTool):
    """Analyze symmetry with LLM summary."""

    name = "LLMStructureSymmetryAnalysis"
    description = (
        "Analyze symmetry properties of a crystal structure with natural language explanation. "
        "Input: CIF file path or structure JSON. "
        "Returns: symmetry data and LLM-generated analysis."
    )

    llm: BaseLanguageModel = None
    llm_chain: LLMChain = None
    base_tool: StructureSymmetryAnalysis = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.base_tool = StructureSymmetryAnalysis()
        self.llm = llm
        prompt = PromptTemplate(
            template=SYMMETRY_ANALYSIS_PROMPT, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, structure_input: str) -> str:
        """Analyze symmetry and get LLM summary."""
        try:
            raw_result = self.base_tool._run(structure_input)
            summary = self.llm_chain.run(data=raw_result)
            return f"Raw Data:\n{raw_result}\n\nLLM Analysis:\n{summary}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, structure_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class LLMStructureProperties(BaseTool):
    """Calculate properties with LLM interpretation."""

    name = "LLMStructureProperties"
    description = (
        "Calculate physical and chemical properties with natural language interpretation. "
        "Input: CIF file path or structure JSON. "
        "Returns: properties data and LLM-generated analysis."
    )

    llm: BaseLanguageModel = None
    llm_chain: LLMChain = None
    base_tool: StructureProperties = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.base_tool = StructureProperties()
        self.llm = llm
        prompt = PromptTemplate(
            template=PROPERTIES_ANALYSIS_PROMPT, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, structure_input: str) -> str:
        """Calculate properties and get LLM interpretation."""
        try:
            raw_result = self.base_tool._run(structure_input)
            summary = self.llm_chain.run(data=raw_result)
            return f"Raw Data:\n{raw_result}\n\nLLM Interpretation:\n{summary}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, structure_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class LLMCompositionAnalysis(BaseTool):
    """Analyze composition with LLM explanation."""

    name = "LLMCompositionAnalysis"
    description = (
        "Analyze chemical composition with natural language explanation. "
        "Input: chemical formula (e.g., 'Li2O', 'SiO2'). "
        "Returns: composition data and LLM-generated explanation."
    )

    llm: BaseLanguageModel = None
    llm_chain: LLMChain = None
    base_tool: CompositionAnalysis = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.base_tool = CompositionAnalysis()
        self.llm = llm
        prompt = PromptTemplate(
            template=COMPOSITION_ANALYSIS_PROMPT, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, formula: str) -> str:
        """Analyze composition and get LLM explanation."""
        try:
            raw_result = self.base_tool._run(formula)
            summary = self.llm_chain.run(data=raw_result)
            return f"Raw Data:\n{raw_result}\n\nLLM Explanation:\n{summary}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, formula: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class LLMStructureComparison(BaseTool):
    """Compare structures with LLM analysis."""

    name = "LLMStructureComparison"
    description = (
        "Compare two crystal structures with natural language analysis. "
        "Input: two CIF file paths separated by '|'. "
        "Returns: comparison data and LLM-generated analysis."
    )

    llm: BaseLanguageModel = None
    llm_chain: LLMChain = None
    base_tool: StructureComparison = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.base_tool = StructureComparison()
        self.llm = llm
        prompt = PromptTemplate(
            template=COMPARISON_ANALYSIS_PROMPT, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, structures_input: str) -> str:
        """Compare structures and get LLM analysis."""
        try:
            raw_result = self.base_tool._run(structures_input)
            summary = self.llm_chain.run(data=raw_result)
            return f"Raw Data:\n{raw_result}\n\nLLM Analysis:\n{summary}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, structures_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class LLMStructureSubstitution(BaseTool):
    """Substitute elements with LLM analysis."""

    name = "LLMStructureSubstitution"
    description = (
        "Substitute elements in crystal structure with natural language analysis. "
        "Input: structure path, old element, new element separated by '|'. "
        "Returns: substitution data and LLM-generated analysis."
    )

    llm: BaseLanguageModel = None
    llm_chain: LLMChain = None
    base_tool: StructureSubstitution = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.base_tool = StructureSubstitution()
        self.llm = llm
        prompt = PromptTemplate(
            template=SUBSTITUTION_ANALYSIS_PROMPT, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, substitution_input: str) -> str:
        """Perform substitution and get LLM analysis."""
        try:
            raw_result = self.base_tool._run(substitution_input)
            summary = self.llm_chain.run(data=raw_result)
            return f"Raw Data:\n{raw_result}\n\nLLM Analysis:\n{summary}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, substitution_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class LLMSurfaceGeneration(BaseTool):
    """Generate surfaces with LLM analysis."""

    name = "LLMSurfaceGeneration"
    description = (
        "Generate surface slabs with natural language analysis. "
        "Input: structure path, Miller indices, slab thickness, vacuum thickness separated by '|'. "
        "Returns: surface data and LLM-generated analysis."
    )

    llm: BaseLanguageModel = None
    llm_chain: LLMChain = None
    base_tool: SurfaceGeneration = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.base_tool = SurfaceGeneration()
        self.llm = llm
        prompt = PromptTemplate(
            template=SURFACE_ANALYSIS_PROMPT, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, surface_input: str) -> str:
        """Generate surface and get LLM analysis."""
        try:
            raw_result = self.base_tool._run(surface_input)
            summary = self.llm_chain.run(data=raw_result)
            return f"Raw Data:\n{raw_result}\n\nLLM Analysis:\n{summary}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, surface_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class LLMCoordinationAnalysis(BaseTool):
    """Analyze coordination with LLM interpretation."""

    name = "LLMCoordinationAnalysis"
    description = (
        "Analyze coordination environments with natural language interpretation. "
        "Input: structure path and optional cutoff radius separated by '|'. "
        "Returns: coordination data and LLM-generated analysis."
    )

    llm: BaseLanguageModel = None
    llm_chain: LLMChain = None
    base_tool: CoordinationAnalysis = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.base_tool = CoordinationAnalysis()
        self.llm = llm
        prompt = PromptTemplate(
            template=COORDINATION_ANALYSIS_PROMPT, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, coordination_input: str) -> str:
        """Analyze coordination and get LLM interpretation."""
        try:
            raw_result = self.base_tool._run(coordination_input)
            # Truncate if too long for LLM
            if len(raw_result) > 3000:
                raw_result = raw_result[:3000] + "\n... (truncated for LLM analysis)"
            summary = self.llm_chain.run(data=raw_result)
            return f"Raw Data:\n{raw_result}\n\nLLM Interpretation:\n{summary}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, coordination_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class LLMLatticeParameterOptimization(BaseTool):
    """Analyze lattice parameters with LLM interpretation."""

    name = "LLMLatticeParameterOptimization"
    description = (
        "Analyze lattice parameters with natural language interpretation. "
        "Input: CIF file path or structure JSON. "
        "Returns: lattice data and LLM-generated analysis."
    )

    llm: BaseLanguageModel = None
    llm_chain: LLMChain = None
    base_tool: LatticeParameterOptimization = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.base_tool = LatticeParameterOptimization()
        self.llm = llm
        prompt = PromptTemplate(
            template=LATTICE_ANALYSIS_PROMPT, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, structure_input: str) -> str:
        """Analyze lattice parameters and get LLM interpretation."""
        try:
            raw_result = self.base_tool._run(structure_input)
            summary = self.llm_chain.run(data=raw_result)
            return f"Raw Data:\n{raw_result}\n\nLLM Analysis:\n{summary}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, structure_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


__all__ = [
    "LLMStructureFromCIF",
    "LLMStructureSymmetryAnalysis",
    "LLMStructureProperties",
    "LLMCompositionAnalysis",
    "LLMStructureComparison",
    "LLMStructureSubstitution",
    "LLMSurfaceGeneration",
    "LLMCoordinationAnalysis",
    "LLMLatticeParameterOptimization",
]
