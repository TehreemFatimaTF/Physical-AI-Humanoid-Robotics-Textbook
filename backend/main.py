from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import cohere
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend_build", html=True), name="frontend")

# ---------------- CONFIG ----------------
QDRANT_URL = "https://your-cluster-url"
QDRANT_API_KEY = "your-qdrant-api-key"
COLLECTION_NAME = "humanoid_ai_book"

cohere_client = cohere.Client("your-cohere-api-key")
EMBED_MODEL = "embed-english-v3.0"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---------------- FASTAPI ----------------
app = FastAPI(title="RAG Chatbot API")

class QuestionInput(BaseModel):
    question: str
    selected_text: str = None

# ---------------- HELPERS ----------------
def embed(text: str):
    """Generate embedding vector from text using Cohere"""
    response = cohere_client.embed(
        model=EMBED_MODEL,
        input_type="search_query",
        texts=[text]
    )
    return response.embeddings[0]

def search_qdrant(query_vector, top_k=3):
    """Search top-k similar chunks in Qdrant"""
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    # Extract text from results
    chunks = [hit.payload["text"] for hit in results]
    return chunks

# ---------------- ENDPOINT ----------------
@app.post("/api/ask")
def ask_book(input: QuestionInput):
    if input.selected_text:
        # If user selected text, use it as context
        context = input.selected_text
    else:
        # Otherwise, search Qdrant for relevant chunks
        query_vec = embed(input.question)
        chunks = search_qdrant(query_vec)
        context = " ".join(chunks)

    # Placeholder for now: integrate LLM to generate answer using context
    answer = f"Question: {input.question}\nContext from book: {context}\nAnswer: [LLM will generate answer here]"

    return {
        "question": input.question,
        "answer": answer
    }
