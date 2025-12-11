import os
import re
import time
import requests
import xml.etree.ElementTree as ET

import langchain
import paperqa
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.embeddings.openai import OpenAIEmbeddings
from pypdf.errors import PdfReadError

# ------------ arXiv API config ------------ #

# Use HTTPS as best practice
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# Shared session with polite User-Agent
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "ChemCrow-ArxivTool/0.1 "
            "(https://example.org; mailto:you@example.org)"
        )
    }
)

# Simple global rate limit for API calls (arXiv ToU: <= 1 request / 3s)
_LAST_API_CALL = 0.0
_API_MIN_INTERVAL = 3.0  # seconds


# ------------ ArXiv helper functions (raw API) ------------ #

def _throttled_arxiv_get(params: dict) -> requests.Response:
    """Call the arXiv export API with a simple 3s throttle."""
    global _LAST_API_CALL
    now = time.time()
    elapsed = now - _LAST_API_CALL
    if elapsed < _API_MIN_INTERVAL:
        time.sleep(_API_MIN_INTERVAL - elapsed)
    _LAST_API_CALL = time.time()

    resp = SESSION.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def _arxiv_api_query(
    search: str,
    max_results: int = 20,
) -> list[dict]:
    """
    Call the arXiv export API directly and parse results.

    Returns a list of dicts with keys:
      'id', 'title', 'authors', 'summary', 'published', 'pdf_url'
    """
    params = {
        # search all fields with the LLM query
        "search_query": f"all:{search}",
        "start": 0,
        "max_results": max_results,
    }

    resp = _throttled_arxiv_get(params)

    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    entries: list[dict] = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        id_el = entry.find("atom:id", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)

        title = (title_el.text or "").strip() if title_el is not None else ""
        entry_id = (id_el.text or "").strip() if id_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        published = (published_el.text or "").strip() if published_el is not None else ""

        # authors
        authors_list = []
        for a in entry.findall("atom:author", ns):
            name_el = a.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors_list.append(name_el.text.strip())
        authors = ", ".join(authors_list)

        # build a pdf URL from the id: http://arxiv.org/abs/... -> http://arxiv.org/pdf/....pdf
        pdf_url = ""
        if "/abs/" in entry_id:
            pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"

        entries.append(
            {
                "id": entry_id,
                "title": title,
                "authors": authors,
                "summary": summary,
                "published": published,
                "pdf_url": pdf_url,
            }
        )

    return entries


def arxiv_scraper(
    search: str,
    pdir: str = "arxiv_query",
    max_results: int = 20,
) -> dict:
    """
    Use the raw arXiv API to find up to max_results papers matching `search`,
    download their PDFs, and return a mapping:

        path -> {"citation": <citation_string>}
    """
    if not os.path.isdir(pdir):
        os.makedirs(pdir, exist_ok=True)

    results = _arxiv_api_query(search, max_results=max_results)
    papers: dict = {}

    for r in results:
        if not r.get("pdf_url"):
            continue

        try:
            # build a reasonably unique filename from the id
            arxiv_id = r["id"].split("/")[-1] if r["id"] else "arxiv"
            filename = f"{arxiv_id}.pdf"
            path = os.path.join(pdir, filename)

            # PDF downloads: sequential, via same SESSION/UA
            pdf_resp = SESSION.get(r["pdf_url"], timeout=30)
            pdf_resp.raise_for_status()

            with open(path, "wb") as f:
                f.write(pdf_resp.content)

            citation = (
                f"Title: {r['title']}\n"
                f"Authors: {r['authors']}\n"
                f"Published: {r['published']}\n"
                f"ArXiv ID: {arxiv_id}\n"
                f"URL: {r['id']}\n\n"
                f"Abstract: {r['summary']}"
            )

            papers[path] = {"citation": citation}
        except Exception:
            # skip any entry that fails download / parse
            continue

    return papers


# ------------ LLM wrapper functions (ChemCrow-style) ------------ #

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
        A search query that would bring up papers that can answer
        this question would be: """,
    )

    query_chain = langchain.chains.llm.LLMChain(llm=llm, prompt=prompt)

    base_dir = "arxiv_query"
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    search = query_chain.run(query).strip()
    print("\nArXiv search:", search)

    # subdirectory per search term (strip whitespace)
    subdir = re.sub(r"\s+", "", search) or "default"
    search_dir = os.path.join(base_dir, subdir)
    if not os.path.isdir(search_dir):
        os.mkdir(search_dir)

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
