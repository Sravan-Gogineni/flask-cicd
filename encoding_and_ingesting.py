from pinecone import Pinecone, ServerlessSpec
import os
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv(override=True)

# === Config ===
filepath = "/home/sravan/Desktop/Projects/NLPPROJECT"
index_name = "nlp-research-project"
dimension = 384  # Embedding size of "all-MiniLM-L6-v2"

# === Initialize Pinecone ===
pc = Pinecone(api_key=os.getenv("apikey"), environment=os.getenv("env"))

# === Load and chunk the text files ===
all_chunks = []
chunk_size = 512

for file in os.listdir(filepath):
    if file.endswith(".txt"):
        with open(os.path.join(filepath, file), "r", encoding="utf-8") as f:
            text = f.read()
            # Chunk by word count
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                all_chunks.append(chunk)

print(f"📄 Total chunks generated: {len(all_chunks)}")

# === Load the embedding model ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# === Create index if it doesn't exist ===
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Index '{index_name}' created successfully")
else:
    print(f"ℹ️ Index '{index_name}' already exists")

# === Connect to index ===
index = pc.Index(index_name)

# === Prepare and upsert vectors with metadata ===
vectors = [
    {
        "id": str(uuid4()),
        "values": embedding.tolist(),
        "metadata": {"text": chunk}
    }
    for chunk, embedding in zip(all_chunks, embeddings)
]

# === Batch upload ===
batch_size = 50
for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading vectors"):
    batch = vectors[i:i + batch_size]
    try:
        index.upsert(vectors=batch)
    except Exception as e:
        print(f"❌ Failed to upsert batch {i // batch_size + 1}: {e}")
    else:
        print(f"✅ Batch {i // batch_size + 1} uploaded")

print("🚀 Ingestion complete!")
