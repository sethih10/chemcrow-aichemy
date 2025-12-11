import os

# Just in case – ChemCrow / LangChain usually wants this set even if we never call the LLM.
os.environ.setdefault("OPENAI_API_KEY", "dummy-key")

from chemcrow.agents import ChemCrow

def get_tools_from_chemcrow(chem_model):
    """
    Try to pull the tools list from various likely places,
    depending on how the ChemCrow class is implemented.
    """
    # 1) Direct attribute on ChemCrow
    if hasattr(chem_model, "tools") and chem_model.tools:
        return chem_model.tools

    # 2) On the agent_executor (LangChain AgentExecutor)
    if hasattr(chem_model, "agent_executor"):
        ae = chem_model.agent_executor
        if hasattr(ae, "tools") and ae.tools:
            return ae.tools

        # 3) On an inner .agent (ChatZeroShotAgent etc.)
        if hasattr(ae, "agent") and hasattr(ae.agent, "tools"):
            return ae.agent.tools

    raise RuntimeError("Couldn't find tools attribute on ChemCrow. "
                       "Check ChemCrow implementation / version.")

def main():
    # Instantiate ChemCrow with your usual settings
    chem_model = ChemCrow(
        model="gpt-4.1-mini",   # or whatever you're using now
        temp=0.1,
        streaming=False,
        verbose=False,
    )

    tools_obj = get_tools_from_chemcrow(chem_model)

    # Tools can be a list or a dict depending on how they're stored
    if isinstance(tools_obj, dict):
        tools_iter = tools_obj.values()
    else:
        tools_iter = tools_obj

    print("\n=== ChemCrow tools (default) ===\n")
    for idx, tool in enumerate(tools_iter, start=1):
        name = getattr(tool, "name", repr(tool))
        desc = getattr(tool, "description", "(no description)")
        print(f"{idx}. {name}")
        print(f"   {desc}")

        # If it's a LangChain `Tool` / `StructuredTool`, you can sometimes inspect args:
        # (won't always be present, so we guard it)
        args_schema = getattr(tool, "args", None) or getattr(tool, "args_schema", None)
        if args_schema is not None:
            print(f"   Args schema: {args_schema}")
        print()

if __name__ == "__main__":
    main()
