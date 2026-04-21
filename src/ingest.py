import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Step 1: Load CSV
df = pd.read_csv("C:/Users/suraj/Desktop/CODING/E-Commerce RAG/data/products.csv")

# Step 2: Convert to documents
documents = []

for _, row in df.iterrows():
    text = f"""
    Product Name: {row['name']}.
    Price: ₹{row['price']}.
    Specifications: {row['specs']}.
    Customer Reviews: {row['reviews']}.
    """

    doc = Document(
        page_content=text.strip(),
        metadata={
            "price": row["price"],
            "name": row["name"]
        }
    )

    documents.append(doc)

# Step 3: Create embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Step 4: Store in Qdrant (local)
client = QdrantClient(path="qdrant_data")

# ✅ Create collection first
if "products" not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name="products",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

# ✅ Then connect
qdrant = Qdrant(
    client=client,
    collection_name="products",
    embeddings=embeddings
)

# ✅ Add data
qdrant.add_documents(documents)

print("✅ Data successfully stored in Qdrant!")