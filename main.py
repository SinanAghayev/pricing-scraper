from typing import Annotated, Sequence, TypedDict
from urllib.parse import urlparse
from dotenv import load_dotenv
import pandas as pd
import requests
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import config


load_dotenv()

websites_tried: list[str] = []
websites_found: list[str] = []

iterations_done_count = 0


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def check_website_exists(url: str) -> bool:
    """Check whether a website exists and responds with a valid HTTP status
    and add the website to the global list if it is not duplicated"""
    print(f"Checking url: {url}")
    try:
        r = requests.get(
            url, timeout=5, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
        )
        global websites_found
        global websites_tried
        url = url.rstrip("/")

        websites_tried.append(url)
        if r.status_code < 400:
            original_domain = urlparse(url).netloc
            final_domain = urlparse(r.url).netloc

            if original_domain != final_domain:
                print(f"Redirected to different domain: {final_domain}")
                return False

            if url in websites_found:
                print(f"Duplicate ignored: {url}")
                return False
            else:
                websites_found.append(url)
            return True
        return False
    except requests.RequestException:
        return False


tools = [check_website_exists]

model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def agent(state: AgentState) -> AgentState:
    global iterations_done_count
    iterations_done_count += 1
    print(f"{iterations_done_count=}")
    system_prompt = SystemMessage(
        content=f"""
            You are Scraper, a research agent.

            Goal:
            - Find websites that have a pricing page.

            Rules:
                1. Generate candidate URL which points to a pricing page.
                2. Call `check_website_exists` on a candidate URL.
                3. Don't try the ones that have been tried before.
                4. Prefer SaaS / software / online services.

                Already found: {websites_found}
                Tried before: {websites_tried}
            """
    )

    response = model.invoke([system_prompt])

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    print(f"{len(websites_found)=}")
    print(f"{len(websites_tried)=}")
    return {"messages": list(state["messages"]) + [response]}


def should_continue(state: AgentState) -> str:
    """Determine the conditional routing"""
    if (
        len(websites_found) > config.WEBSITE_COUNT_NEEDED
        or iterations_done_count >= config.MAX_ITERATIONS
    ):
        return "end"

    return "continue"


def print_messages(messages):
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {"continue": "agent", "end": END})

app = graph.compile()


def run_agent():
    print("\n ===== BEGIN =====")

    state = {"messages": []}
    for step in app.stream(
        state, stream_mode="values", config={"recursion_limit": config.MAX_ITERATIONS}
    ):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== FINISHED =====")


def write_websites_to_file(websites, filename="websites.xlsx"):
    df = pd.DataFrame(websites, columns=["website"])
    df.to_excel(filename, index=False)


if __name__ == "__main__":
    run_agent()
    print(websites_found)
    write_websites_to_file(websites_found)
