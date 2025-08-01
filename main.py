from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from retrieve import search
from dotenv import load_dotenv
from openai import OpenAI

# Load .env variables
load_dotenv()

# Get Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize OpenAI-compatible Groq client
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://utano-health-assistant.vercel.app",  # ✅ No trailing slash
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QuestionRequest(BaseModel):
    question: str

# Root route
@app.get("/")
def root():
    return {"message": "Utano Health Assistant Backend is running!"}

# ✅ Test route to check backend is live
@app.get("/ping")
def ping():
    return {"message": "Backend is alive!"}

# Main API route
@app.post("/api/answer")
async def get_answer(req: QuestionRequest):
    question = req.question
    print(f"📩 Received question: {question}")

    if not groq_api_key:
        print("❌ GROQ_API_KEY is missing")
        return {"answer": "Backend misconfiguration: GROQ_API_KEY is missing."}

    # Step 1: Retrieve context
    try:
        retrieved_chunks = search(question)
        context = "\n\n".join(retrieved_chunks)
        print(f"📚 Retrieved context: {context[:200]}...")  # Show preview
    except Exception as e:
        print("❌ Error retrieving chunks:", str(e))
        return {"answer": "Error retrieving information from the database."}

    # Step 2: Generate answer
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant answering health-related questions using only the given context. Make it compassionate and helpful. Your name is Utano. Display bullet points in a list form."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.7,
            max_tokens=300,
        )
        answer = response.choices[0].message.content
        print("✅ Answer generated successfully")
        return {"answer": answer}
    except Exception as e:
        print("❌ Groq error:", str(e))
        return {"answer": "An error occurred while generating the answer from the model."}
