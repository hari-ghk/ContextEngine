from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from db import collection, model
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain.tools import tool
import os
import requests
from tavily import TavilyClient

load_dotenv()
app = FastAPI()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

llm = ChatGroq(
    model="llama-3.1-8b-instant",
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

tools = [search_python_docs, search_web]
agent = create_react_agent(llm, tools)

@app.post("/agent")
async def run_agent(request: AgentRequest):
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": request.user_query}],
        })
        return {"reply": result["messages"][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

