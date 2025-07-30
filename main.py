
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from retrieve import search
from dotenv import load_dotenv
from openai import OpenAI

# Load .env variables (including GROQ_API_KEY)
load_dotenv()

# Get Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize OpenAI client with Groq base URL
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (allow all origins — change in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request
class QuestionRequest(BaseModel):
    question: str

# API endpoint
@app.post("/api/answer")
async def get_answer(req: QuestionRequest):
    question = req.question
    print(f"Received question: {question}")

    # Step 1: Retrieve relevant chunks from FAISS
    try:
        retrieved_chunks = search(question)
        context = "\n\n".join(retrieved_chunks)
    except Exception as e:
        print("Error retrieving chunks:", str(e))
        return {"answer": "Error retrieving information from the database."}

    # Step 2: Generate answer using Groq (OpenAI-compatible client)
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Best general-purpose model on Groq
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant answering health-related questions using only the given context.Make it compassionate and helpful.Your name is Utano.Display bullet points in a list form"
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
        return {"answer": answer}
    except Exception as e:
        print("Groq error:", str(e))
        return {"answer": "An error occurred while generating the answer from the model."}
