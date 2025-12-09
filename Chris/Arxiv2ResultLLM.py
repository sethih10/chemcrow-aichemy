import os
import re
import requests

import arxiv
import langchain
import paperqa
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.embeddings.openai import OpenAIEmbeddings
from pypdf.errors import PdfReadError


# ------------ ArXiv helper functions ------------ #

def arxiv_scraper(
    search: str,
    pdir: str = "arxiv_query",
    max_results: int = 20,
    timeout: int = 30,
) -> dict:
    """
    Search arxiv.org and download PDFs for the top `max_results` hits
    into directory `pdir`.

    Returns:
        dict[path] = {"citation": <citation_string>}
    """
    # Make a base directory if it doesn't exist
    os.makedirs(pdir, exist_ok=True)

    search_obj = arxiv.Search(
        query=search,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = {}
    for result in search_obj.results():
        try:
            pdf_url = result.pdf_url
            arxiv_id = result.entry_id.split("/")[-1]
            # Each query gets its own sub-folder, so we don't collide too much
            filename = f"{arxiv_id}.pdf"
            path = os.path.join(pdir, filename)

            # Download PDF
            r = requests.get(pdf_url, timeout=timeout)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)

            authors = ", ".join([a.name for a in result.authors])
            published = result.published.isoformat() if hasattr(result, "published") else ""
            citation = (
                f"Title: {result.title}\n"
                f"Authors: {authors}\n"
                f"Published: {published}\n"
                f"ArXiv ID: {arxiv_id}\n"
                f"URL: {result.entry_id}\n\n"
                f"Abstract: {result.summary}"
            )

            papers[path] = {"citation": citation}
        except Exception:
            # Silently skip any failed downloads / parse issues
            continue

    return papers


def arxiv_paper_search(llm, query, max_results=20):
    """
    Use an LLM to compress the user query to a short arXiv search string,
    then run an arXiv search and download the PDFs.
    """
    prompt = langchain.prompts.PromptTemplate(
        input_variables=["question"],
        template="""
        I would like to find scholarly papers to answer
        this question: {question}. Your response must be at
        most 10 words long.
        'A search query that would bring up papers that can answer
        this question would be: '""",
    )

    query_chain = langchain.chains.llm.LLMChain(llm=llm, prompt=prompt)

    # Base dir to keep all arxiv queries
    base_dir = "arxiv_query"
    os.makedirs(base_dir, exist_ok=True)

    search = query_chain.run(query)
    print("\nArXiv search:", search)

    # Make a subdir per search term (remove spaces)
    subdir = re.sub(r"\s+", "", search)
    search_dir = os.path.join(base_dir, subdir)
    os.makedirs(search_dir, exist_ok=True)

    papers = arxiv_scraper(search, pdir=search_dir, max_results=max_results)
    return papers


def arxiv2result_llm(
    llm,
    query,
    k: int = 5,
    max_sources: int = 2,
    openai_api_key: str = None,
    max_results: int = 20,
):
    """
    ArXiv-based analogue of scholar2result_llm:
    - Use LLM to generate a focused search query
    - Search arXiv & download PDFs
    - Use paperqa to answer the question from those PDFs
    """
    papers = arxiv_paper_search(llm, query, max_results=max_results)
    if len(papers) == 0:
        return "Not enough arXiv papers found"

    docs = paperqa.Docs(
        llm=llm,
        summary_llm=llm,
        embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key),
    )

    not_loaded = 0
    for path, data in papers.items():
        try:
            docs.add(path, data["citation"])
        except (ValueError, FileNotFoundError, PdfReadError):
            not_loaded += 1

    if not_loaded > 0:
        print(
            f"\nFound {len(papers.items())} arXiv papers "
            f"but couldn't load {not_loaded}."
        )
    else:
        print(f"\nFound {len(papers.items())} arXiv papers and loaded all of them.")

    answer = docs.query(query, k=k, max_sources=max_sources).formatted_answer
    return answer


# ------------ ChemCrow tool wrapper ------------ #

class Arxiv2ResultLLM(BaseTool):
    """
    ChemCrow tool for answering technical questions using arxiv.org papers.

    Usage pattern mirrors the existing Semantic Scholar-based LiteratureSearch tool.
    """

    name = "ArxivLiteratureSearch"
    description = (
        "Useful to answer questions that require technical knowledge, "
        "by searching arxiv.org (physics, CS, math, etc.) and reading "
        "the top papers. Ask a specific question."
    )

    llm: BaseLanguageModel = None
    openai_api_key: str = None
    max_results: int = 20

    def __init__(
        self,
        llm: BaseLanguageModel,
        openai_api_key: str = None,
        max_results: int = 20,
    ):
        super().__init__()
        self.llm = llm
        self.openai_api_key = openai_api_key
        self.max_results = max_results

    def _run(self, query: str) -> str:
        return arxiv2result_llm(
            self.llm,
            query,
            openai_api_key=self.openai_api_key,
            max_results=self.max_results,
        )

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
