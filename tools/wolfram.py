import os
import logging
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

def wolfram_compute(query: str) -> str:
    appid = os.getenv("WOLFRAM_APPID")
    if not appid:
        logger.warning("WOLFRAM_APPID not set. Skipping Wolfram query.")
        return "Wolfram APPID not set."
    logger.info(f"Querying WolframAlpha: {query}")
    params = urlencode({"i": query, "appid": appid})
    url = f"https://api.wolframalpha.com/v1/result?{params}"
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            return r.text.strip()
        logger.error(f"Wolfram error {r.status_code}: {r.text}")
        return f"Wolfram error: {r.status_code}"
    except Exception as e:
        logger.error(f"Wolfram request failed: {e}")
        return f"Wolfram error: {e}"
