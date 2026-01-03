"""
Optimized RAG Pipeline (No repeated chunking/embedding)
"""

# ================================
# 🧩 IMPORTS
# ================================
import os
import joblib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

# ================================
# 🧩 ENV + CLIENTS
# ================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = st.secrets["GROQ_MODEL"]

groq_client = Groq(api_key=GROQ_API_KEY)

import streamlit as st


# ================================
# 🧩 CONSTANTS
# ================================
TEXT_FILE = "text.txt"
JOBLIB_FILE = "text_data.joblib"
COLLECTION_NAME = "rajiv_dixit_texts"

# ================================
# 🧩 STEP 1 — TEXT LOAD (ONE TIME)
# ================================
def load_text():
    if not os.path.exists(JOBLIB_FILE):
        with open(TEXT_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        joblib.dump(text, JOBLIB_FILE)
    else:
        text = joblib.load(JOBLIB_FILE)

    return text


# ================================
# 🧩 STEP 2 — CHUNKING (ONE TIME)
# ================================
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# ================================
# 🧩 STEP 3 — EMBEDDING MODEL (ONE TIME)
# ================================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ================================
# 🧩 STEP 4 — CHROMADB INIT (ONE TIME)
# ================================
chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chroma_store",
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ================================
# 🧩 STEP 5 — INGEST DATA (ONLY IF EMPTY)
# ================================
def ingest_data():
    if collection.count() > 0:
        return  # already ingested

    text = load_text()
    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks)

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )


# 🚀 Run ingestion ONCE at startup
ingest_data()


# ================================
# 🧩 SIMILARITY SEARCH
# ================================
def search_similar_chunks(query, top_k=10):
    query_embedding = embedding_model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results["documents"][0]


# ================================
# 🧩 GROQ LLM
# ================================
def generate_answer(context, query):
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ],
        temperature=0.2,
        max_tokens=512
    )
    return completion.choices[0].message.content


# ================================
# 🧩 MAIN (ONLY QUERY LOGIC)
# ================================
def main(query):
    top_chunks = search_similar_chunks(query)

    prompt = f"""
Use ONLY the following context to answer:

{chr(10).join(top_chunks)}

Rules:
- Hinglish query → Hinglish answer
- English query → English answer
- If not found user query related data in context  → say "I don't have sufficient data"
- Be concise and human-friendly
- only give answer related to your context dont add your additional knowledge if not found in context 
"""

    return generate_answer(prompt, query)
