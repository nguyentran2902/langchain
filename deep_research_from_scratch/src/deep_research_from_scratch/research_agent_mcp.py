
"""Research Agent with MCP Integration.

This module implements a research agent that integrates with Model Context Protocol (MCP)
servers to access tools and resources. The agent demonstrates how to use MCP filesystem
server for local document research and analysis.

Key features:
- MCP server integration for tool access
- Async operations for concurrent tool execution (required by MCP protocol)
- Filesystem operations for local document research
- Secure directory access with permission checking
- Research compression for efficient processing
- Lazy MCP client initialization for LangGraph Platform compatibility
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

from typing_extensions import Literal
from langchain_openai import ChatOpenAI

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END

from deep_research_from_scratch.prompts import research_agent_prompt_with_mcp, compress_research_system_prompt, compress_research_human_message
from deep_research_from_scratch.state_research import ResearcherState, ResearcherOutputState
from deep_research_from_scratch.utils import get_today_str, think_tool, get_current_dir

# ===== CONFIGURATION =====
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

# MCP server configuration for filesystem access
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # Auto-install if needed
            "@modelcontextprotocol/server-filesystem",
            str(get_current_dir() / "files") # Path to research documents directory
        ],
        "transport": "stdio"  # Communication via stdin/stdout
    }
}

# Global client variable - will be initialized lazily
_client = None

def get_mcp_client():
    """Get or initialize MCP client lazily to avoid issues with LangGraph Platform."""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client

# Initialize models
compress_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=CUSTOM_BASE_URL,
) 
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=CUSTOM_BASE_URL,
) 

# ===== MCP TOOL WRAPPERS =====

@tool
async def read_file(path: str) -> str:
    """Reads the contents of a file from the local filesystem."""
    client = get_mcp_client()
    # The MCP filesystem server exposes a 'read_file' tool.
    # We find it and invoke it.
    mcp_tools = await client.get_tools()
    read_tool = next((t for t in mcp_tools if t.name == 'read_file'), None)
    if read_tool:
        return await read_tool.ainvoke({"path": path})
    return "Error: read_file tool not found on MCP server."

@tool
async def list_files(path: str = ".") -> str:
    """Lists the files in a directory on the local filesystem."""
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    list_tool = next((t for t in mcp_tools if t.name == 'list_files'), None)
    if list_tool:
        file_list = await list_tool.ainvoke({"path": path})
        return "\n".join(file_list)
    return "Error: list_files tool not found on MCP server."

# ===== AGENT NODES =====

async def llm_call(state: ResearcherState):
    """Analyze current state and decide on tool usage with MCP integration.

    This node:
    1. Retrieves available tools from MCP server
    2. Binds tools to the language model
    3. Processes user input and decides on tool usage

    Returns updated state with model response.
    """
    # Use our wrapped MCP tools + the standard think_tool
    tools = [read_file, list_files, think_tool]

    # Initialize model with tool binding
    model_with_tools = model.bind_tools(tools)

    # Process user input with system prompt
    response = {
        "researcher_messages": [
            await model_with_tools.ainvoke(
                [SystemMessage(content=research_agent_prompt_with_mcp.format(date=get_today_str()))] +
                state["researcher_messages"]
            )
        ]
    }
    return response

async def tool_node(state: ResearcherState):
    """Execute tool calls using MCP tools.

    This node:
    1. Retrieves current tool calls from the last message
    2. Executes all tool calls using async operations (required for MCP)
    3. Returns formatted tool results

    Note: MCP requires async operations due to inter-process communication
    with the MCP server subprocess. This is unavoidable.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    async def execute_tools():
        """Execute all tool calls. MCP tools require async execution."""
        tools = [read_file, list_files, think_tool]
        tools_by_name = {tool.name: tool for tool in tools}

        # Execute tool calls (sequentially for reliability)
        observations = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            # Our wrapped tools are async, so we always use ainvoke
            observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)

        # Format results as tool messages
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for observation, tool_call in zip(observations, tool_calls)
        ]

        return tool_outputs

    messages = await execute_tools()

    return {"researcher_messages": messages}

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for further processing or reporting.

    This function filters out think_tool calls and focuses on substantive
    file-based research content from MCP tools.
    """

    # Filter out messages with tool_calls and ToolMessages, as the compression model doesn't expect them
    # Only check for tool_calls attribute on messages that have it (AIMessage)
    from langchain_core.messages import ToolMessage
    
    filtered_messages = [
        m for m in state.get("researcher_messages", []) 
        if not (hasattr(m, 'tool_calls') and m.tool_calls) and not isinstance(m, ToolMessage)
    ]

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages_for_compression = [SystemMessage(content=system_message)] + filtered_messages + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages_for_compression)

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

max_iterations = 5  # Set a hard limit on the number of research loops

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue with tool execution or compress research.

    Determines whether to continue with tool execution or compress research
    based on whether the LLM made tool calls.
    """
    # Check if we have exceeded the maximum number of iterations
    if len(state['researcher_messages']) > max_iterations * 2:
        return "compress_research"

    messages = state["researcher_messages"]
    last_message = messages[-1]

    # Continue to tool execution if tools were called
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, compress research findings
    return "compress_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder_mcp = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder_mcp.add_node("llm_call", llm_call)
agent_builder_mcp.add_node("tool_node", tool_node)
agent_builder_mcp.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder_mcp.add_edge(START, "llm_call")
agent_builder_mcp.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",        # Continue to tool execution
        "compress_research": "compress_research",  # Compress research findings
    },
)
agent_builder_mcp.add_edge("tool_node", "llm_call")  # Loop back for more processing
agent_builder_mcp.add_edge("compress_research", END)

# Compile the agent
agent_mcp = agent_builder_mcp.compile()

# ===== MAIN EXECUTION =====

async def main():
    """Main execution function to run the MCP research agent."""
    # Initialize the client once
    get_mcp_client()
    # Define the research topic
    research_topic = "Summarize the contents of the local research documents, focusing on the key findings from each file."

    # Set up the initial state
    initial_state = {
        "researcher_messages": [HumanMessage(content=research_topic)],
        "research_topic": research_topic
    }

    print(f"Starting MCP research for topic: '{research_topic}'\n")
    
    # Invoke the agent
    final_state = await agent_mcp.ainvoke(initial_state)

    print("\n\n===== COMPRESSED RESEARCH (MCP) =====\n")
    print(final_state["compressed_research"])

if __name__ == "__main__":
    asyncio.run(main())
