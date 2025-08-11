import logging
from typing import List
from openai import OpenAI
from agent.state import AgentState, Evidence
from tools.search import tavily_search
from tools.wiki import wikipedia_lookup
from tools.wolfram import wolfram_compute
from tools.notes import notes_search

logger = logging.getLogger(__name__)
client = OpenAI()

def _llm(prompt: str, temperature: float = 0.2) -> str:
    logger.debug(f"Sending prompt to LLM (temp={temperature}): {prompt[:120]}...")
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = r.choices[0].message.content or ""
    logger.debug(f"LLM response: {text[:120]}...")
    return text

def plan_node(state: AgentState) -> AgentState:
    logger.info("Planning steps...")
    prompt = f'''You are a step planner. For the question: "{state["question"]}"
From the tools [Search, Wikipedia, Math, Notes] propose MAX 3 steps.
Return ONLY a JSON array of strings, e.g. ["Search","Wikipedia","Answer"].'''
    text = _llm(prompt, 0.1).strip()
    try:
        import json
        plan: List[str] = json.loads(text)
    except Exception as e:
        logger.warning(f"Failed to parse plan JSON: {e}. Using default.")
        plan = ["Search", "Answer"]
    logger.info(f"Plan created: {plan}")
    return {**state, "plan": plan}

def execute_node(state: AgentState) -> AgentState:
    plan = state.get("plan") or []
    if not plan:
        logger.warning("No plan found in state. Skipping execute.")
        return state

    next_step, *rest = plan
    logger.info(f"Executing tool: {next_step}")
    out: List[Evidence] = []

    step = next_step.lower()
    if step == "search":
        results = tavily_search(state["question"])
        out = [{"source": r["url"], "content": r["snippet"]} for r in results]
    elif step == "wikipedia":
        page = wikipedia_lookup(state["question"])
        out = [{"source": page["url"], "content": page["summary"]}]
    elif step == "math":
        ans = wolfram_compute(state["question"])
        out = [{"source": "WolframAlpha", "content": ans}]
    elif step == "notes":
        hits = notes_search(state["question"], k=4)
        out = [{"source": f'notes:{h["id"]}', "content": h["text"]} for h in hits]
    elif step == "answer":
        logger.debug("Reached 'answer' step — clearing plan to finish.")
        return {
            **state,
            "plan": [],
            "scratchpad": [*state.get("scratchpad", []), "Reached Answer"],
            "evidence": [*state.get("evidence", [])],
        }

    logger.debug(f"Tool output: {out[:2]}{'...' if len(out) > 2 else ''}")
    return {
        **state,
        "plan": rest if rest else ["Answer"],
        "scratchpad": [*state.get("scratchpad", []), f"Executed {next_step}"],
        "evidence": [*state.get("evidence", []), *out],
    }


def aggregate_node(state: AgentState) -> AgentState:
    logger.info("Aggregating evidence...")
    items = state.get("evidence", [])
    context = "\n".join([f'[{i+1}] {e["content"]} (src: {e["source"]})' for i, e in enumerate(items)])
    summary = _llm(f"From the following evidence, produce a concise summary:\n{context}", 0.3)
    new_ev: Evidence = {"source": "aggregate", "content": summary}
    return {**state, "evidence": [*items, new_ev], "scratchpad": [*state.get("scratchpad", []), "Aggregated"]}

def reflect_node(state: AgentState) -> AgentState:
    logger.info("Reflecting on completeness...")
    prompt = f"""Question: {state['question']}
We have {len(state.get('evidence', []))} evidence items.
Is the answer likely complete? Reply ONLY "OK" or "NO – add Search/Wikipedia/Math/Notes"."""
    verdict = (_llm(prompt, 0).strip() or "OK").upper()
    logger.info(f"Reflection verdict: {verdict}")
    if verdict.startswith("NO"):
        tool = "Search"
        if "WIKIPEDIA" in verdict:
            tool = "Wikipedia"
        elif "MATH" in verdict:
            tool = "Math"
        elif "NOTES" in verdict:
            tool = "Notes"
        return {**state, "plan": [tool, "Answer"]}
    return state

def answer_node(state: AgentState) -> AgentState:
    logger.info("Generating final answer...")
    ev = [e for e in state.get("evidence", []) if e["source"] != "aggregate"]
    citations = "\n".join([f'[{i+1}] {e["source"]}' for i, e in enumerate(ev[:5])])
    context = "\n---\n".join(e["content"] for e in state.get("evidence", []))
    ans = _llm(f"""Question: {state['question']}\nContext:\n{context}\n\nWrite a clear answer (~10 sentences max).""", 0.4)
    ans = ans + ("\n\nSources:\n" + (citations or "—"))
    return {**state, "answer": ans}
