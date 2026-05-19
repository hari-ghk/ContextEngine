from db import model, collection
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class Query(BaseModel):
    user_query: str

@app.post("/query")
def query_documentation(query: Query):
    embedded_user_query = model.encode(query.user_query).tolist()
    results = collection.query(
        query_embeddings=[embedded_user_query],
        n_results=5
    )
    chunks = results["documents"][0]
    combined_context = "\n\n".join(chunks)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant who'll help answer the query based only on this documentation:\n\n {combined_context}"},
            {"role": "user", "content": query.user_query}
        ]
    )
    return response.choices[0].message.content


