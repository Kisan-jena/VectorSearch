# 🎬 Movie Recommendations with Vector Search & RAG

A semantic search system for movie recommendations using **MongoDB Atlas Vector Search**, **Hugging Face Embeddings**, and **RAG (Retrieval-Augmented Generation)** principles.

---

## 📋 Table of Contents

| Section | Description |
|---------|-------------|
| [✨ Features](#-features) | Key capabilities of the system |
| [🔧 Prerequisites](#-prerequisites) | Requirements before starting |
| [🚀 Setup Instructions](#-setup-instructions) | Step-by-step setup guide |
| [🔍 MongoDB Vector Search Index](#-mongodb-atlas-vector-search-index-setup) | Create the vector search index |
| [▶️ Running the Project](#️-running-the-project) | How to run and test |
| [📚 How It Works](#-how-it-works-explanation) | Technical explanation |
| [📁 Project Structure](#-project-structure) | File organization |
| [🔧 Troubleshooting](#-troubleshooting) | Common issues and solutions |

---

## ✨ Features

- **Semantic Search**: Search movies by meaning, not just keywords
- **Vector Embeddings**: Uses Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model
- **MongoDB Atlas Integration**: Leverages MongoDB's vector search capabilities
- **384-Dimensional Vectors**: Represents movie plots as numerical embeddings
- **Cosine Similarity**: Finds semantically similar movies

---

## 🔧 Prerequisites

Before you begin, ensure you have:

| Requirement | Description |
|------------|-------------|
| **Python 3.13+** | Required Python version |
| **[uv](https://docs.astral.sh/uv/)** | Package manager for running scripts |
| **MongoDB Atlas Account** | Free tier (M0) works perfectly |
| **Hugging Face Account** | For free API token |

---

## 🚀 Setup Instructions

### **Step 1: Install `uv` Package Manager**

```sh
# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### **Step 2: Clone/Download the Project**

```sh
cd Project1
```

### **Step 3: Create MongoDB Atlas Cluster**

1. Go to [MongoDB Atlas](https://cloud.mongodb.com)
2. Sign up/Login
3. Create a **FREE M0 cluster**
4. Load **Sample Datasets** → Choose `sample_mflix`
5. Create a **Database User** with password
6. Add your **IP Address** to whitelist (or `0.0.0.0/0` for development)
7. Get your **Connection String**: `mongodb+srv://username:password@cluster.mongodb.net/`

### **Step 4: Get Hugging Face API Token**

1. Go to [Hugging Face](https://huggingface.co)
2. Sign up/Login
3. Navigate to **Settings** → **Access Tokens**
4. Create a **new token** (Read access is sufficient)
5. Copy the token (starts with `hf_...`)

### **Step 5: Create Environment File**

Create a `.env` file in the `Project1` directory:

```sh
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
HF_TOKEN=hf_your_huggingface_token_here
```

⚠️ **Important**: Never commit `.env` to Git! It's already in `.gitignore`.

---

## 🔍 MongoDB Atlas Vector Search Index Setup

> **⚠️ CRITICAL**: This step is required for vector search to work!

### **Step 1: Navigate to Atlas Search**

1. In MongoDB Atlas, go to your cluster
2. Click **"Browse Collections"**
3. Select `sample_mflix` database → `movies` collection
4. Click **"Search Indexes"** tab (or "Search & Vector Search")

### **Step 2: Create Vector Search Index**

1. Click **"Create Search Index"**
2. Select **"Vector Search"**
3. Choose **"JSON Editor"**
4. Set **Index Name**: `PlotSemanticSearch`
5. Select **Database**: `sample_mflix`
6. Select **Collection**: `movies`
7. Click **"Next"**

### **Step 3: Configure Index (JSON)**

Paste this exact configuration:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "plot_embedding_hf",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

**Configuration Breakdown:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `type` | `"vector"` | This is a vector search index |
| `path` | `"plot_embedding_hf"` | Field containing embeddings |
| `numDimensions` | `384` | Vector size (model output) |
| `similarity` | `"cosine"` | Similarity metric used |

### **Step 4: Create and Wait**

1. Click **"Create Search Index"**
2. Wait for status → **"READY"** (1-2 minutes)
3. Verify: **21,349 documents indexed** ✅

---

## ▶️ Running the Project

### **Step 1: Generate Embeddings (First Time Only)**

Uncomment lines 67-80 in `movie_recs.py`:

```python
print("\nGenerating embeddings for movies...")
count = 0
for doc in collection.find({'plot':{"$exists": True}}).limit(50):
  try:
    doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
    collection.replace_one({'_id': doc['_id']}, doc)
    count += 1
    print(f"[{count}/50] Generated embedding for: {doc['title']}")
    time.sleep(0.5)
  except Exception as e:
    print(f"Error: {e}")
    break
```

Run the script:

```sh
uv run movie_recs.py
```

**⚠️ Comment out these lines after first run to avoid regenerating embeddings.**

### **Step 2: Test Vector Search**

Edit the `query` variable in `movie_recs.py`:

```python
query = "imaginary characters from outer space at war"
```

Run the script:

```sh
uv run movie_recs.py
```

### **Try Different Queries:**

| Query Type | Example Query |
|------------|--------------|
| **Sci-Fi War** | `"imaginary characters from outer space at war"` |
| **Romance** | `"a romantic love story with heartbreak"` |
| **Adventure** | `"epic adventure with treasure hunting"` |
| **Mystery** | `"detective solving mysterious crimes"` |
| **Comedy** | `"funny family comedy with kids"` |

---

## 📚 How It Works (Explanation)

### **1. Vector Embeddings Concept**

**What are embeddings?**
- Text → **384 numbers** (a vector)
- Similar meanings → Similar numbers
- Enables semantic search

**Example:**
```
"Luke Skywalker joins forces with a Jedi Knight..."
        ↓ (AI Model: all-MiniLM-L6-v2)
[0.023, -0.123, 0.567, ..., 0.091]  ← 384 numbers
```

### **2. Semantic Search Process**

```
┌─────────────────────────────────────────┐
│ Step 1: Query → Embedding               │
│ "space war" → [0.12, 0.45, -0.33, ...]  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Step 2: Compare with Database           │
│ MongoDB compares query vector with      │
│ all movie vectors using cosine          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Step 3: Return Top Matches              │
│ Movies with highest similarity scores   │
└─────────────────────────────────────────┘
```

### **3. Cosine Similarity**

```python
similarity = (A · B) / (||A|| × ||B||)
```

| Score | Meaning |
|-------|---------|
| **1.0** | Identical meaning ✅ |
| **0.5** | Somewhat similar |
| **0.0** | No relation |
| **-1.0** | Opposite meaning ❌ |

### **4. Vector Search vs Keyword Search**

| Search Type | Query: "space war" | Results |
|-------------|-------------------|---------|
| **Keyword** | Matches exact words | Only: "space" AND "war" |
| **Vector** | Understands meaning | Star Wars, Galaxy battles, Alien invasions, etc. |

### **5. Code Flow Breakdown**

```python
# 1️⃣ Load credentials from .env
mongo_uri = os.getenv("MONGODB_URL")
hf_token = os.getenv("HF_TOKEN")

# 2️⃣ Connect to MongoDB Atlas
client = pymongo.MongoClient(mongo_uri)
db = client.sample_mflix
collection = db.movies

# 3️⃣ Create Hugging Face API client
hf_client = InferenceClient(token=hf_token)

# 4️⃣ Generate embedding function
def generate_embedding(text: str) -> list[float]:
    result = hf_client.feature_extraction(
        text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return result.tolist()

# 5️⃣ Convert query to vector
query = "space war"
query_embedding = generate_embedding(query)  # [0.12, 0.45, ...]

# 6️⃣ Vector search using MongoDB
results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": query_embedding,      # Your 384 numbers
    "path": "plot_embedding_hf",        # Compare with this field
    "numCandidates": 100,               # Check top 100
    "limit": 4,                         # Return top 4
    "index": "PlotSemanticSearch",      # Use this index
  }}
])

# 7️⃣ Display results
for movie in results:
    print(movie["title"], movie["plot"])
```

### **6. Key Components**

| Component | Purpose |
|-----------|---------|
| **Hugging Face API** | Converts text to embeddings |
| **all-MiniLM-L6-v2** | AI model (384-dimensional) |
| **MongoDB Atlas** | Stores vectors & performs search |
| **PlotSemanticSearch** | Vector search index name |
| **plot_embedding_hf** | Field containing vectors |
| **Cosine Similarity** | Measures vector similarity |

---

## 📁 Project Structure

```
Project1/
├── movie_recs.py          # Main script with inline dependencies
├── README.md              # This file
├── .env                   # Environment variables (⚠️ NOT in Git)
└── .env.example           # Template for environment file
```

### **Dependencies (Auto-managed by `uv`)**

The script uses [PEP 723](https://peps.python.org/pep-0723/) inline metadata:

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pymongo",           # MongoDB driver
#     "python-dotenv",     # Load environment variables
#     "requests",          # HTTP client
#     "huggingface-hub",   # Hugging Face API
#     "numpy",             # Numerical operations
# ]
# ///
```

**No need for separate `requirements.txt` or `pyproject.toml`!** ✨

---

## 🔧 Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `MONGODB_URL not found` | Missing `.env` file | Create `.env` in Project1 directory |
| `400 Bad Request (HF)` | Invalid HF token | Check token has read permissions |
| `No results found` | Embeddings not generated | Run Step 1 first, uncomment embedding code |
| `Index not found` | Vector index missing | Create `PlotSemanticSearch` index in Atlas |
| `Slow performance` | Rate limiting | Add `time.sleep(0.5)` between requests |

### **Still Having Issues?**

1. Verify MongoDB connection string format
2. Ensure vector search index status is **READY**
3. Check that embeddings exist in database (query one document)
4. Confirm index name matches: `PlotSemanticSearch`

---

## 🎯 Next Steps

- [ ] Generate embeddings for more movies (increase `limit(50)` → `limit(1000)`)
- [ ] Try different embedding models
- [ ] Implement filtering by genre, year, rating
- [ ] Add a web interface (Flask/FastAPI)
- [ ] Combine with LLM for conversational recommendations
- [ ] Deploy to production

---

## 📖 Resources

| Resource | Link |
|----------|------|
| **MongoDB Vector Search Docs** | [Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/) |
| **Hugging Face Inference API** | [Documentation](https://huggingface.co/docs/api-inference/index) |
| **Sentence Transformers** | [Website](https://www.sbert.net/) |
| **Tutorial Video** | [YouTube](https://www.youtube.com/watch?v=JEBDfGqrAUA) |

---

## 📝 License

This project is for educational purposes.

---

<div align="center">

**Happy Searching! 🎬🔍**

Made with ❤️ using MongoDB Atlas, Hugging Face & Python

</div>
