from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent import search_web
from db import collection, model
from dotenv import load_dotenv
import os
import time
from evals import evaluate_faithfulness, evaluate_answer_relevance



load_dotenv()
app = FastAPI()

llm = ChatAnthropic(
    model=os.getenv("ANTHROPIC_MODEL"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0
)

class AgentRequest(BaseModel):
    user_query: str

class AgentResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str
    tokens_used: int
    latency_ms: int
    cost_usd: float
    faithfulness_score: float | None = None
    relevance_score: float | None = None


class RunContext:
    def __init__(self):
        self.sources: list[str] = []
        self.tokens_used: int = 0
        self.chunks : str = ""

@app.post("/ask")
async def ask_claude(request: AgentRequest):
    start = time.time()
    context = RunContext()

    #Search the RAG for python docs based on the user input
    @tool
    def search_python_docs(query: str):
        """Search Python documentation. Use this while answering Python questions."""
        embedding = model.encode(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results = 3
        )
        context.sources.extend([m["source"] for m in results["metadatas"][0]])
        context.chunks = "\n\n".join(results["documents"][0])
        return "\n\n".join(results["documents"][0])


    try:
        mcp_client = MultiServerMCPClient(
            {
                "python-repl":{
                    "url": "http://127.0.0.1:8000/mcp",
                    "transport": "streamable-http"
                }

            }
        )
        mcp_tools = await mcp_client.get_tools()
        all_tools = [search_python_docs, search_web] + mcp_tools
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
        last_message = result["messages"][-1]
        end = time.time()
        input_cost = (last_message.usage_metadata["input_tokens"] / 1_000_000) * 1.0
        output_cost = (last_message.usage_metadata["output_tokens"]/ 1_000_000) * 5.0
        total_cost = input_cost + output_cost
        latency_ms = int((end-start) * 1000)

        #calculate faithfulness only when it's a rag retrieval
        if context.chunks:
            faith = evaluate_faithfulness(request.user_query, context.chunks, last_message)
            faithfulness_score = faith["score"]
        relevance = evaluate_answer_relevance(request.user_query, last_message)
        relevance_score = relevance["score"]
        return AgentResponse(
            answer=last_message.content,
            sources=list(set(context.sources)),
            model=os.getenv("ANTHROPIC_MODEL"),
            tokens_used=last_message.usage_metadata["total_tokens"],
            latency_ms=latency_ms,
            cost_usd=total_cost,
            faithfulness_score=faithfulness_score,
            relevance_score=relevance_score,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


