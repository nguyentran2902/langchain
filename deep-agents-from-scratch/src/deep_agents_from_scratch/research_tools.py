"""Research Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""
import os
from datetime import datetime
import uuid, base64
import sys
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

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated, Literal

from deep_agents_from_scratch.prompts import SUMMARIZE_WEB_SEARCH
from deep_agents_from_scratch.state import DeepAgentState


# ===== CONFIGURATION =====
try:
    import nest_asyncio
    # Apply nest_asyncio if no event loop is running, which is common in scripts
    # and can be necessary for some environments like Jupyter.
    # It allows the asyncio event loop to be nested, which is required by some
    # environments where an event loop is already running (like Jupyter or the langgraph server).
    nest_asyncio.apply()
except (ImportError, RuntimeError):
    pass  # nest_asyncio not installed or loop is already running in a way that can't be patched.

CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

# Summarization model 
summarization_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=CUSTOM_BASE_URL,
) 
tavily_client = TavilyClient()

class Summary(BaseModel):
    """Schema for webpage content summarization."""
    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    if os.name == "nt":  # Windows
        return datetime.now().strftime("%a %b %#d, %Y")
    else:  # Linux/macOS
        return datetime.now().strftime("%a %b %-d, %Y")
def run_tavily_search(
    search_query: str, 
    max_results: int = 1, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> dict:
    """Perform search using Tavily API for a single query.

    Args:
        search_query: Search query to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        Search results dictionary
    """
    result = tavily_client.search(
        search_query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic
    )

    return result

def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Summary object with filename and summary
    """
    try:
        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(Summary)

        # Generate summary
        summary_and_filename = structured_model.invoke([
            HumanMessage(content=SUMMARIZE_WEB_SEARCH.format(
                webpage_content=webpage_content, 
                date=get_today_str()
            ))
        ])

        return summary_and_filename

    except Exception:
        # Return a basic summary object on failure
        return Summary(
            filename="search_result.md",
            summary=webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
        )

def process_search_results(results: dict) -> list[dict]:
    """Process search results by summarizing content where available.

    Args:
        results: Tavily search results dictionary

    Returns:
        List of processed results with summaries
    """
    processed_results = []

    # Create a client for HTTP requests.
    # `trust_env=False` prevents the client from using system-wide proxy settings
    # (like HTTP_PROXY), which can interfere with fetching public URLs.
    HTTPX_CLIENT = httpx.Client(trust_env=False)

    for result in results.get('results', []):

        # Get url 
        url = result['url']

        # Read url
        response = HTTPX_CLIENT.get(url)

        if response.status_code == 200:
            # Convert HTML to markdown
            raw_content = markdownify(response.text)
            summary_obj = summarize_webpage_content(raw_content)
        else:
            # Use Tavily's generated summary
            raw_content = result.get('raw_content', '')
            summary_obj = Summary(
                filename="URL_error.md",
                summary=result.get('content', 'Error reading URL; try another search.')
            )

        # uniquify file names
        uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = f"{name}_{uid}{ext}"

        processed_results.append({
            'url': result['url'],
            'title': result['title'],
            'summary': summary_obj.summary,
            'filename': summary_obj.filename,
            'raw_content': raw_content,
        })

    return processed_results

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
    """Search web and save detailed results to files while returning minimal context.

    Performs web search and saves full content to files for context offloading.
    Returns only essential information to help the agent decide on next steps.

    Args:
        query: Search query to execute
        state: Injected agent state for file storage
        tool_call_id: Injected tool call identifier
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Command that saves full results to files and provides minimal summary
    """
    # Execute search
    search_results = run_tavily_search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    ) 

    # Process and summarize results
    processed_results = process_search_results(search_results)

    # Save each result to a file and prepare summary
    files = state.get("files", {})
    saved_files = []
    summaries = []

    for i, result in enumerate(processed_results):
        # Use the AI-generated filename from summarization
        filename = result['filename']

        # Create file content with full details
        file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {query}
**Date:** {get_today_str()}

## Summary
{result['summary']}

## Raw Content
{result['raw_content'] if result['raw_content'] else 'No raw content available'}
"""

        files[filename] = file_content
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}...")

    # Create minimal summary for tool message - focus on what was collected
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""

    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(summary_text, tool_call_id=tool_call_id)
            ],
        }
    )

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    - How complex is the question: Have I reached the number of search limits?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

async def main():
    """Main execution function to test the research tools."""
    # Define a sample research topic/query
    research_query = "What are the latest advancements in large language models in 2024?"

    # Set up an initial state, simulating the agent's state
    # The `tavily_search` tool expects a state dictionary and a tool_call_id
    initial_state: DeepAgentState = {
        "messages": [HumanMessage(content=research_query)],
        "files": {},
    }

    print(f"Starting tool test for query: '{research_query}'\n")
    
    # To test a tool with InjectedState and InjectedToolCallId, we must simulate
    # a ToolCall object and pass the state via the config.
    # For direct invocation testing, all arguments, including injected ones,
    # should be placed in the 'args' dictionary.
    tool_call_input = {
        "name": "tavily_search",
        "args": {"query": research_query, "state": initial_state},
        "id": "test_tool_call",
        "type": "tool_call",
    }
    
    command_output = tavily_search.invoke(tool_call_input)

    print("\n\n===== TOOL EXECUTION RESULT =====\n")
    print(command_output)

if __name__ == "__main__":
    # This allows running the file directly to test the main function
    asyncio.run(main())
