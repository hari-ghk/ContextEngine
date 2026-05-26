from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from db import collection, model
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain.tools import tool
import os
from tavily import TavilyClient
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AIMessage


load_dotenv()
app = FastAPI()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

class AgentRequest(BaseModel):
    user_query: str

@tool
def search_python_docs(query:str) -> str:
    """Search the Python documentation for information about a query.
    Use this when answering questions about Python."""
    print(f"🔍 RAG tool called with: {query}")
    embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )
    chunks = results["documents"][0]
    print(f"📄 RAG returned: {chunks[:200]}...")
    return "\n\n".join(chunks)

@tool
def search_web(query:str) -> str:
    """Search the web for current information that is not available
    in the Python documentation. Use this as a fallback when the
    docs don't contain the answer."""

    results = tavily.search(query=query, max_results=3)
    contents = [result["content"] for result in results["results"]]
    return "\n\n".join(contents)

@app.post("/agent")
async def run_agent(request: AgentRequest):
    tools = [search_python_docs, search_web]
    agent = create_react_agent(llm, tools)
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": request.user_query}],
        })
        return {"reply": result["messages"][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/mcp")
async def run_agent_with_mcp(request: AgentRequest):
    try:
            mcp_client = MultiServerMCPClient(
                {
                    "python-repl": {
                        "url": "http://localhost:8000/mcp",
                        "transport": "streamable_http"
                    }
                }
            )
            mcp_tools = await mcp_client.get_tools()
            print(f"🔧 MCP tools loaded: {[t.name for t in mcp_tools]}")
            all_tools = [search_python_docs, search_web] + mcp_tools
            print(f"🔧 All tools: {[t.name for t in all_tools]}")
            agent = create_react_agent(llm,
                                       all_tools,
                                       prompt="""You are a helpful Python assistant with access to tools.

                                       Guidelines:
                                       - When asked to write or execute code, always use execute_python_code tool
                                       - Always include actual tool output in your response
                                       - For general questions, use search_python_docs first
                                       - For current information, use search_web
                                       - Be concise and accurate"""
                                       )
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": request.user_query}],
            })
            ai_replies = [
                msg.content for msg in result["messages"]
                if isinstance(msg, AIMessage) and msg.content
            ]

            if ai_replies:
                return {"reply": ai_replies[-1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


