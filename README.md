# ⚡ High-Performance Semantic Search Engine with FAISS & Sentence-Transformers

🌐 **Idiomas / Languages:** [🇪🇸 Versión en Español](README.es.md) | [🇬🇧 English Version](README.md)


![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![HuggingFace](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-ready implementation of dense vector search and semantic similarity indexing using **Meta AI's FAISS (Facebook AI Similarity Search)** and pre-trained Transformer embeddings.

---

## 📌 Architecture Overview

Traditional lexical search algorithms (e.g., BM25, TF-IDF) rely on exact keyword matching, failing to capture semantic nuance, synonyms, and context. This project builds a dense vector index pipeline:

```
[Text Ingestion] ➡️ [Transformer Embeddings] ➡️ [FP32 Normalization] ➡️ [FAISS IndexFlatL2] ➡️ [Sub-millisecond Retrieval]
```

1. **Text Embedding Generation:** Leverages `all-MiniLM-L6-v2` from Sentence-Transformers to map text into a continuous 384-dimensional vector space.
2. **Dense Indexing:** Uses `faiss.IndexFlatL2` for exact, brute-force Euclidean distance ($L_2$) calculations, optimized in C++ for maximum throughput.
3. **Real-time Query Resolution:** Vectorizes user queries on the fly, computes nearest neighbors ($k$-NN), and returns semantic matches with sub-millisecond latency.

---

## 🚀 Key Features

* **Sub-millisecond Search Latency:** Exact $L_2$ neighbor lookup taking $< 5\text{ ms}$ on local compute.
* **Dense Embedding Integration:** Supports state-of-the-art multi-lingual and domain-specific Transformer models.
* **Strict Memory Footprint:** FP32 vector casting to ensure native FAISS C++ memory alignment and avoid memory fragmentation.
* **Interactive CLI Interface:** Real-time query execution with performance metrics (distance score and execution time).

---

## 🛠️ Tech Stack & Requirements

* **Python 3.10+**
* **FAISS-CPU** (`faiss-cpu==1.8.0` or `faiss-gpu`)
* **Sentence-Transformers** (`sentence-transformers>=2.2.0`)
* **NumPy** (`numpy>=1.24.0`)
* **PyTorch**

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/JulioRubio6767/faiss-semantic-search-engine.git
cd faiss-semantic-search-engine
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate     # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

Run the search engine:
```bash
python main.py
```

### Example Query Execution:
```text
--- SISTEMA DE BÚSQUEDA SEMÁNTICA (FAISS 1.8) ---
Vectorizando base de conocimiento...
Índice listo con 5 frases cargadas.

------------------------------------------------------------
¿Qué quieres buscar?: computación del futuro y física
Buscando en la base vectorial...

RESULTADO:
-> Los algoritmos cuánticos prometen cambiar la forma en que resolvemos problemas complejos.
Distancia: 0.8412 | Tiempo: 2.14 ms
```

---

## 📈 Scalability Roadmap (From Prototype to Enterprise RAG)

* [ ] Add hierarchical indexing (`faiss.IndexIVFFlat`) for clustering and million-scale scaling.
* [ ] Replace CLI with a high-throughput **FastAPI** REST interface.
* [ ] Integrate document chunking pipelines for PDF/TXT ingestion (RecursiveCharacterTextSplitter).
* [ ] Support cosine similarity via normalized inner product (`faiss.IndexFlatIP`).

---

## 👤 Author
**Julio Andrés Rubio Echeverría**  
*Analytics & Machine Learning Engineer*  
* [LinkedIn](https://www.linkedin.com/in/juliorubio-data/)
* [GitHub](https://github.com/JulioRubio6767)
