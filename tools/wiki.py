import logging
import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)

def wikipedia_lookup(query: str):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"
    logger.info(f"Looking up Wikipedia for: {query}")
    try:
        r = requests.get(url, timeout=10)
        if not r.ok:
            logger.warning(f"Wikipedia returned {r.status_code} for query: {query}")
            return {"url": "https://wikipedia.org", "summary": "No article"}
        data = r.json()
        page_url = (data.get("content_urls", {}).get("desktop", {}) or {}).get("page") or data.get("url") or "https://wikipedia.org"
        return {"url": page_url, "summary": data.get("extract", "")}
    except Exception as e:
        logger.error(f"Wikipedia lookup failed: {e}")
        return {"url": "https://wikipedia.org", "summary": "No article"}
