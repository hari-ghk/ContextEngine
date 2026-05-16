from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import os

load_dotenv()
app = FastAPI()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    system_message: str
    user_message: str
    message_history: list[Message]

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": request.system_message},
                *[{"role": m.role, "content": m.content} for m in request.message_history],
                {"role": "user","content": request.user_message}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))