import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI

from tools import search_tool, weather_info_tool, hub_stats_tool, news_tool
from retriever import guest_info_tool

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_tool, weather_info_tool, hub_stats_tool, news_tool, guest_info_tool]
chat_with_tools = llm.bind_tools(tools)



# Create the AgentState and AgentGraph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return{
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

# Create the AgentGraph
builder = StateGraph(AgentState)

# Define the nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))


# Define the edges: determine how the control flow moves between nodes
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # if the last message is a tool call, route to the tools node
    # Otherwise, provide a direct response
    tools_condition,
)

builder.add_edge("tools", "assistant")
alfred = builder.compile()

if __name__ == "__main__":
    response = alfred.invoke(
        {
            "messages": [HumanMessage(content="Ai là Facebook và model phổ biến nhất của họ là gì?. Và cho tôi biết thời tiết hiện tại ở Hà Nội")],
        }
    )
    print("🎩 Alfred's Response:")
    print(response["messages"][-1].content)

    # Save visualization of the graph
    image_data = alfred.get_graph().draw_mermaid_png()

    # Lưu vào file
    with open("alfred_graph.png", "wb") as f:
        f.write(image_data)


