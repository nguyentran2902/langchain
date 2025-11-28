
"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

import sys
import os
import asyncio
from dotenv import load_dotenv
# --- Thêm src vào sys.path để import deep_research_from_scratch ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project_root/src
SRC_DIR = BASE_DIR  # src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Load .env từ project_root ---
PROJECT_ROOT = os.path.dirname(SRC_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

print("OPENAI_API_KEY =", os.environ.get("OPENAI_API_KEY"))

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from deep_research_from_scratch.utils import get_today_str
from deep_research_from_scratch.prompts import final_report_generation_prompt
from deep_research_from_scratch.state_scope import AgentState, AgentInputState
from deep_research_from_scratch.research_agent_scope import clarify_with_user, write_research_brief
from deep_research_from_scratch.multi_agent_supervisor import supervisor_agent
import os

CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

# ===== Config =====

try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it


writer_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=CUSTOM_BASE_URL,
    # Alternative if above doesn't work:
    # base_url=CUSTOM_BASE_URL,
) 

# ===== FINAL REPORT GENERATION =====

from deep_research_from_scratch.state_scope import AgentState

async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }

# ===== GRAPH CONSTRUCTION =====
# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the full workflow
agent = deep_researcher_builder.compile()

# ===== MAIN EXECUTION =====

async def main():
    """Main execution function to run the full research agent."""
    
    # Define the initial user query
    user_query = "What are the latest trends in AI-powered drug discovery, focusing on small molecule generation and protein folding?"

    # Set up the initial state
    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }

    print(f"Starting full research process for: '{user_query}'\n")
    
    # Invoke the agent
    final_state = await agent.ainvoke(initial_state)

    print(f"\n\n===== FINAL REPORT =====\n\n{final_state['final_report']}")

if __name__ == "__main__":
    asyncio.run(main())
