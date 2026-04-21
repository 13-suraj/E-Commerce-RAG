import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

from google import genai

# -----------------------------
# 1. Load API key
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# -----------------------------
# 2. Setup Gemini client (NEW API)
# -----------------------------
genai_client = genai.Client(api_key=api_key)
model_name = "gemini-2.5-flash"

# -----------------------------
# 3. Load embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------------
# 4. Load Qdrant DB (LOCAL)
# -----------------------------
qdrant_client = QdrantClient(path="qdrant_data")

qdrant = Qdrant(
    client=qdrant_client,
    collection_name="products",
    embeddings=embeddings   # IMPORTANT (not embeddings=)
)

# -----------------------------
# 5. Create retriever
# -----------------------------
retriever = qdrant.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# 6. Take user input
# -----------------------------
query = input("🔍 Ask your product question: ")

# -----------------------------
# 7. Retrieve relevant products
# -----------------------------
docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in docs])

# -----------------------------
# 8. Create prompt
# -----------------------------
prompt = f"""
You are an expert shopping assistant.

User Query:
{query}

Here are some relevant products:
{context}

Compare them and recommend the best option.
Explain clearly why it is best.
"""

# -----------------------------
# 9. Generate response (NEW API)
# -----------------------------
response = genai_client.models.generate_content(
    model=model_name,
    contents=prompt
)

# -----------------------------
# 10. Print result
# -----------------------------
print("\n🧠 AI Recommendation:\n")
print(response.text)