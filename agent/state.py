from typing import List, Optional, TypedDict

class Evidence(TypedDict):
    source: str
    content: str

class AgentState(TypedDict, total=False):
    question: str
    plan: List[str]
    scratchpad: List[str]
    evidence: List[Evidence]
    answer: Optional[str]
