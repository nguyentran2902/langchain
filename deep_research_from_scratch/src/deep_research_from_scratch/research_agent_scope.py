
"""User Clarification and Research Brief Generation.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research.
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
from datetime import datetime
from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from deep_research_from_scratch.state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState
import os

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


# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    if os.name == "nt":  # Windows
        return datetime.now().strftime("%a %b %#d, %Y")
    else:  # Linux/macOS
        return datetime.now().strftime("%a %b %-d, %Y")

CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# ===== WORKFLOW NODES =====
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=CUSTOM_BASE_URL,
    # Alternative if above doesn't work:
    # base_url=CUSTOM_BASE_URL,
) 
 
def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])

    # Route based on clarification need
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

# ===== GRAPH CONSTRUCTION =====

# Build the scoping workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

# Compile the workflow
scope_research = deep_researcher_builder.compile()

# ===== MAIN EXECUTION =====


async def main():
    """Main execution function to run the research scoping agent."""
    
    # Define the initial user query
    user_query = "Tell me about the latest in AI for drug discovery."

    # Set up the initial state
    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }

    print(f"Starting research scoping for: '{user_query}'\n")
    
    # Invoke the agent
    final_state = await scope_research.ainvoke(initial_state)

    # Check the output and print accordingly
    if final_state.get("research_brief"):
        print("===== RESEARCH BRIEF GENERATED =====\n")
        print(final_state["research_brief"])
    else:
        print("===== CLARIFICATION NEEDED =====\n")
        print(final_state["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
