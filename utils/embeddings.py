from typing import List
from openai import OpenAI

_client = OpenAI()

def embed(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Return embeddings for a list of texts."""
    resp = _client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def cosine(a: list, b: list) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(x*x for x in b) ** 0.5
    return dot / (na * nb + 1e-10)
