from typing import Any

from langgraph.graph import END, StateGraph

from grounded_rag.agent.config import AgentConfig
from grounded_rag.agent.nodes import AgentNodes
from grounded_rag.agent.state import AgentState

# Enough for retrieve/grade/rewrite repeated to the configured limit, plus generation,
# one repair and finalisation, with room to spare.
RECURSION_LIMIT = 50


def build_graph(nodes: AgentNodes, config: AgentConfig) -> Any:
    """Wire the loop.

    retrieve, grade, then either rewrite and retrieve again or generate, verify, optionally
    repair and verify again, and finalize. Any step can end in `abstain`, and every cycle is
    bounded by `config`.
    """

    def after_retrieve(state: AgentState) -> str:
        # Nothing readable matched at all, so rewriting the query cannot help.
        return "abstain" if state.stop_reason or not state.context else "grade"

    def after_grade(state: AgentState) -> str:
        if state.stop_reason or state.grade is None:
            return "abstain"
        if state.grade.sufficient:
            return "generate"
        return "rewrite" if state.rewrites_used < config.max_rewrites else "abstain"

    def after_rewrite(state: AgentState) -> str:
        return "abstain" if state.stalled or state.stop_reason else "retrieve"

    def after_generate(state: AgentState) -> str:
        generated = state.generated
        if state.stop_reason or generated is None:
            return "abstain"
        return "verify" if generated.answerable and generated.claims else "abstain"

    def after_verify(state: AgentState) -> str:
        if state.reports[-1].all_verified:
            return "finalize"
        return "repair" if state.repairs_used < config.max_repairs else "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify", nodes.verify)
    graph.add_node("repair", nodes.repair)
    graph.add_node("finalize", nodes.finalize)
    graph.add_node("abstain", nodes.abstain)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, ["grade", "abstain"])
    graph.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "abstain"])
    graph.add_conditional_edges("rewrite", after_rewrite, ["retrieve", "abstain"])
    graph.add_conditional_edges("generate", after_generate, ["verify", "abstain"])
    graph.add_conditional_edges("verify", after_verify, ["finalize", "repair"])
    graph.add_edge("repair", "verify")
    graph.add_edge("finalize", END)
    graph.add_edge("abstain", END)
    return graph.compile()


async def run_graph(compiled: Any, state: AgentState) -> AgentState:
    final = await compiled.ainvoke(state, {"recursion_limit": RECURSION_LIMIT})
    return final if isinstance(final, AgentState) else AgentState(**final)
