from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode
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

CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

# Initialize the model with custom endpoint
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_base=CUSTOM_BASE_URL,
    openai_api_key=OPENAI_API_KEY,
    # Alternative if above doesn't work:
    # base_url=CUSTOM_BASE_URL,
)

tools = [write_email]
model_with_tools = llm.bind_tools(tools)

def call_llm(state: MessagesState) -> MessagesState:
    """Run LLM with tool binding"""
    output = model_with_tools.invoke(state["messages"])
    return {"messages": [output]}

# Use prebuilt ToolNode instead of manual implementation
tool_node = ToolNode(tools)

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Route to tool handler or end based on tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
    return END

# Build the graph
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("call_llm", call_llm)
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges(
    "call_llm",
    should_continue,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", END)

# Compile the graph
app = workflow.compile()

# Example usage
if __name__ == "__main__":
    result = app.invoke({
        "messages": [
            ("user", "Send an email to nguyentv14@fpt.com about the meeting tomorrow")
        ]
    })
    
    print("\n=== Final Messages ===")
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content if hasattr(msg, 'content') else msg}")
