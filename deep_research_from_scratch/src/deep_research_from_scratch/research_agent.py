
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

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langgraph.graph import START, END, StateGraph
from typing_extensions import Literal

from deep_research_from_scratch.prompts import (
    compress_research_human_message,
    compress_research_system_prompt,
    research_agent_prompt,
)
from deep_research_from_scratch.state_research import (
    ResearcherOutputState,
    ResearcherState,
)
from deep_research_from_scratch.utils import get_today_str, tavily_search, think_tool
import os
from langchain_openai import ChatOpenAI

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


CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Set up tools and model binding
tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_base=CUSTOM_BASE_URL,
    openai_api_key=OPENAI_API_KEY,
)

model_with_tools = model.bind_tools(tools)

compress_model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_base=CUSTOM_BASE_URL,
    openai_api_key=OPENAI_API_KEY,
)

# ===== AGENT NODES =====



def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }

def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we have a final answer
    return "compress_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()

# ===== MAIN EXECUTION =====

async def main():
    """Main execution function to run a single research agent."""
    
    # Define the research topic
    research_topic = "What are the most promising new applications of Large Language Models (LLMs) in finance?"

    # Set up the initial state
    initial_state = {
        "researcher_messages": [HumanMessage(content=research_topic)],
        "research_topic": research_topic
    }

    print(f"Starting research for topic: '{research_topic}'\n")
    
    # Invoke the agent
    final_state = await researcher_agent.ainvoke(initial_state)

    print("\n\n===== COMPRESSED RESEARCH =====\n")
    print(final_state["compressed_research"])

if __name__ == "__main__":
    asyncio.run(main())
