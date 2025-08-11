from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import plan_node, execute_node, aggregate_node, reflect_node, answer_node

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("aggregate", aggregate_node)
    g.add_node("reflect", reflect_node)
    g.add_node("answer", answer_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "aggregate")
    g.add_edge("aggregate", "reflect")

    # If reflection requests more info, loop to execute; otherwise go to answer
    def router(state: AgentState):
        plan = state.get("plan") or []
        return "execute" if state.get("plan") else "answer"

    g.add_conditional_edges("reflect", router, {"execute": "execute", "answer": "answer"})
    g.add_edge("answer", END)

    return g.compile()
