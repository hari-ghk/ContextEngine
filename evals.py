from anthropic import Anthropic
import json
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def evaluate_faithfulness(question: str, context: str, answer: str) -> dict:
    prompt = f"""You are an evaluation judge.

Question: {question}
Context: {context}
Answer: {answer}

Is every claim in the answer supported by the context?
Return ONLY valid JSON, no other text:
{{"score": 0.0, "reason": "your reason"}}

score 1.0 = fully grounded, Score 0.0 = hallucinated
"""

    response = client.messages.create(
        model = os.getenv("ANTHROPIC_MODEL"),
        max_tokens=200,
        messages = [{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def evaluate_answer_relevance(question: str, answer: str) -> dict:
    prompt = f"""You are an evaluation judge.

Question: {question}
Answer: {answer}

Does this answer directly address the question?
Return ONLY valid JSON, no other text:
{{"score": 0.0, "reason": "your reason"}}
score 1.0 = perfectly relevant, Score 0.0 = completely off topic
"""
    response = client.messages.create(
        model = os.getenv("ANTHROPIC_MODEL"),
        max_tokens=200,
        messages = [{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

if __name__ == "__main__":
    question = "What are Python generators?"
    context = "Generators are functions that use the yield keyword to produce values lazily instead of returning them all at once."
    answer = "Python generators are functions that use yield to produce values one at a time, making them memory efficient."

    faithfulness = evaluate_faithfulness(question, context, answer)
    relevance = evaluate_answer_relevance(question, answer)

    print(f"Faithfulness: {faithfulness}")
    print(f"Answer Relevance: {relevance}")