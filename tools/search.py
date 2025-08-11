import os
import logging
from tavily import TavilyClient

logger = logging.getLogger(__name__)

def tavily_search(query: str, max_results: int = 5):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set. Skipping web search.")
        return []
    logger.info(f"Tavily search for: {query}")
    client = TavilyClient(api_key=api_key)
    res = client.search(query=query, max_results=max_results)
    results = [{"url": r["url"], "snippet": r.get("content", "")} for r in res.get("results", [])]
    logger.debug(f"Tavily returned {len(results)} results.")
    return results
