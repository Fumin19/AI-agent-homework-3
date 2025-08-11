import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from utils.embeddings import embed, cosine

logger = logging.getLogger(__name__)
DATA_PATH = Path("data/notes.json")
EMB_MODEL = "text-embedding-3-small"

def _load_notes() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        logger.warning(f"Notes file {DATA_PATH} does not exist.")
        return []
    logger.debug(f"Loading notes from {DATA_PATH}")
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))

def _save_notes(notes: List[Dict[str, Any]]):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug(f"Saved {len(notes)} notes to {DATA_PATH}")

def _ensure_embeddings(notes: List[Dict[str, Any]]):
    to_embed = [n["text"] for n in notes if not n.get("embedding")]
    if not to_embed:
        return notes
    logger.info(f"Embedding {len(to_embed)} notes...")
    vecs = embed(to_embed, model=EMB_MODEL)
    it = iter(vecs)
    for n in notes:
        if not n.get("embedding"):
            n["embedding"] = next(it)
    _save_notes(notes)
    return notes

def notes_search(query: str, k: int = 4) -> List[Dict[str, Any]]:
    notes = _load_notes()
    if not notes:
        logger.warning("No notes available for search.")
        return []
    notes = _ensure_embeddings(notes)
    qvec = embed([query], model=EMB_MODEL)[0]
    scored = [{"id": n["id"], "text": n["text"], "score": cosine(n["embedding"], qvec)} for n in notes]
    scored.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Found {len(scored)} notes, returning top {k}")
    return scored[:k]
