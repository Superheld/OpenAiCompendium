# Search Systems: Information Retrieval

## 🎯 Ziel
Entwickle intelligente Suchsysteme von klassischer Keyword-Suche bis zu modernen semantischen AI-basierten Ansätzen - das Fundament für alle Information Retrieval Anwendungen.

## 📖 Geschichte & Kontext

**Das Problem:** Menschen suchen Information, aber Computer verstehen Anfragen nicht wie Menschen.

**Evolution der Suche:**
- **1990er**: Boolean Search (AND, OR, NOT)
- **2000er**: PageRank & TF-IDF (Google Revolution)
- **2010er**: Machine Learning Features & Personalization
- **2020er**: Neural Search & Semantic Understanding

**Warum moderne Search wichtig ist:**
- Klassische Keyword-Suche findet nur exakte Matches
- Menschen formulieren Suchanfragen anders als Dokumente geschrieben sind
- Synonyme, Paraphrasen, konzeptuelle Ähnlichkeit werden übersehen
- Kontext und Intent sind wichtiger als exakte Wörter

## 🧮 Konzept & Theorie

### Search System Typen

**1. Keyword-basierte Suche (Classical)**
```python
Query: "Laborkühlschrank"
Matches: Dokumente die exakt "Laborkühlschrank" enthalten
Missed: "Medikamentenkühlschrank", "Apotheker-Kühlgerät"
```

**2. Semantische Suche (Modern)**
```python
Query: "Laborkühlschrank"
Matches: Semantisch ähnliche Konzepte
- "Medikamentenkühlschrank"
- "Pharma-Kühlgerät"
- "Impfstoff-Lagerung"
```

**3. Hybrid Search (Best of Both)**
```python
Query: "Laborkühlschrank Modell ABC-123"
Semantic: Ähnliche Kühlschränke finden
Keyword: Exakte Modellnummer finden
Combined: Optimale Ergebnisse
```

### Die Search Pipeline

**Phase 1: Indexing (Offline)**
```
Dokumente → Text Extraction → Chunking → Embeddings → Index Storage
```

**Phase 2: Query Processing (Online)**
```
User Query → Query Analysis → Search Execution → Ranking → Result Presentation
```

## 🛠️ Implementation

### 1. Classical Search (BM25)

**Warum BM25?**
- Industriestandard für Keyword-Suche
- Schnell und effizient
- Baseline für alle anderen Ansätze

```python
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Dokumente
documents = [
    "Der Laborkühlschrank HMFvh 4001 hat 8 Schubfächer",
    "Medikamentenkühlschrank für Apotheken und Krankenhäuser",
    "Gefrierschrank -80°C für Impfstoff-Lagerung",
    "Standard Haushaltskühlschrank mit Gefrierfach"
]

# Preprocessing
stop_words = set(stopwords.words('german'))

def preprocess(text):
    tokens = word_tokenize(text.lower(), language='german')
    return [token for token in tokens if token.isalnum() and token not in stop_words]

tokenized_docs = [preprocess(doc) for doc in documents]

# BM25 Index erstellen
bm25 = BM25Okapi(tokenized_docs)

# Suche
query = "Kühlschrank für Medikamente"
tokenized_query = preprocess(query)
scores = bm25.get_scores(tokenized_query)

# Top-K Ergebnisse
top_k = 3
top_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]

for doc, score in top_docs:
    print(f"Score: {score:.2f} - {doc}")
```

### 2. Semantic Search (Dense Embeddings)

**Warum Dense Embeddings?**
- Versteht Synonyme und Paraphrasen
- Konzeptuelle Ähnlichkeit
- Sprachübergreifend möglich

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Model laden (Deutsch optimiert)
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Dokumente embedden (mit E5 Prefix)
documents = [
    "Der Laborkühlschrank HMFvh 4001 hat 8 Schubfächer",
    "Medikamentenkühlschrank für Apotheken und Krankenhäuser",
    "Gefrierschrank -80°C für Impfstoff-Lagerung",
    "Standard Haushaltskühlschrank mit Gefrierfach"
]

doc_texts = ["passage: " + doc for doc in documents]
doc_embeddings = model.encode(doc_texts, normalize_embeddings=True)

def semantic_search(query, top_k=3):
    # Query embedden
    query_text = f"query: {query}"
    query_embedding = model.encode(query_text, normalize_embeddings=True)

    # Cosine Similarity berechnen
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Top-K Ergebnisse
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'score': similarities[idx],
            'index': idx
        })

    return results

# Suche
query = "Kühlgerät für Arzneimittel"
results = semantic_search(query)

for result in results:
    print(f"Score: {result['score']:.3f} - {result['document']}")
```

### 3. Hybrid Search (Dense + Sparse)

**Warum Hybrid?**
- Dense: Semantische Ähnlichkeit
- Sparse: Exakte Keywords (IDs, Namen)
- Fusion: Beste Ergebnisse

```python
def hybrid_search(query, top_k=3, alpha=0.7):
    """
    Hybrid search combining BM25 and semantic search
    alpha: Weight for semantic search (1-alpha for BM25)
    """

    # BM25 Search
    tokenized_query = preprocess(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores
    bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-6)

    # Semantic Search
    semantic_results = semantic_search(query, top_k=len(documents))
    semantic_scores = np.array([r['score'] for r in semantic_results])

    # Reorder semantic scores to match document order
    semantic_scores_ordered = np.zeros(len(documents))
    for result in semantic_results:
        semantic_scores_ordered[result['index']] = result['score']

    # Fusion: Weighted combination
    hybrid_scores = alpha * semantic_scores_ordered + (1 - alpha) * bm25_scores

    # Top-K Results
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'hybrid_score': hybrid_scores[idx],
            'semantic_score': semantic_scores_ordered[idx],
            'bm25_score': bm25_scores[idx]
        })

    return results

# Test verschiedene Queries
test_queries = [
    "Laborkühlschrank",  # Exakt (BM25 sollte gut sein)
    "Kühlgerät für Medikamente",  # Semantisch (Dense sollte gut sein)
    "HMFvh 4001"  # Modellnummer (BM25 sollte exakt finden)
]

for query in test_queries:
    print(f"\n=== Query: '{query}' ===")
    results = hybrid_search(query, alpha=0.6)

    for i, result in enumerate(results, 1):
        print(f"{i}. Hybrid: {result['hybrid_score']:.3f} | "
              f"Semantic: {result['semantic_score']:.3f} | "
              f"BM25: {result['bm25_score']:.3f}")
        print(f"   {result['document']}")
```

### 4. Production Search System

```python
import chromadb
from typing import List, Dict, Any

class ProductionSearchSystem:
    def __init__(self, model_name='intfloat/multilingual-e5-large'):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client()
        self.collection = None
        self.bm25 = None

    def create_index(self, documents: List[str], collection_name: str = "search_index"):
        """Create searchable index from documents"""

        # Vector Index (ChromaDB)
        self.collection = self.client.create_collection(collection_name)

        # Generate embeddings
        doc_texts = [f"passage: {doc}" for doc in documents]
        embeddings = self.model.encode(doc_texts, normalize_embeddings=True)

        # Store in ChromaDB
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=[{"doc_id": i} for i in range(len(documents))],
            ids=[f"doc_{i}" for i in range(len(documents))]
        )

        # BM25 Index
        tokenized_docs = [preprocess(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"Index created with {len(documents)} documents")

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid", alpha: float = 0.7):
        """
        Search documents
        mode: 'semantic', 'keyword', 'hybrid'
        alpha: Weight for semantic in hybrid (0=only keyword, 1=only semantic)
        """

        if mode == "semantic":
            return self._semantic_search(query, top_k)
        elif mode == "keyword":
            return self._keyword_search(query, top_k)
        elif mode == "hybrid":
            return self._hybrid_search(query, top_k, alpha)
        else:
            raise ValueError("Mode must be 'semantic', 'keyword', or 'hybrid'")

    def _semantic_search(self, query: str, top_k: int):
        """Pure semantic search using embeddings"""
        query_text = f"query: {query}"
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        return [
            {
                'document': doc,
                'score': 1 - distance,  # Convert distance to similarity
                'method': 'semantic'
            }
            for doc, distance in zip(results['documents'][0], results['distances'][0])
        ]

    def _keyword_search(self, query: str, top_k: int):
        """BM25 keyword search"""
        tokenized_query = preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k documents
        doc_scores = [(self.collection.get()['documents'][i], scores[i])
                     for i in range(len(scores))]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                'document': doc,
                'score': score,
                'method': 'keyword'
            }
            for doc, score in doc_scores[:top_k]
        ]

    def _hybrid_search(self, query: str, top_k: int, alpha: float):
        """Hybrid search combining semantic and keyword"""
        # Implementation would combine both methods
        # For brevity, simplified version
        semantic_results = self._semantic_search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)

        # Simple fusion (real implementation would be more sophisticated)
        combined_results = semantic_results[:top_k]

        return combined_results

# Usage Example
search_system = ProductionSearchSystem()

# Index documents
documents = [
    "Der Laborkühlschrank HMFvh 4001 hat 8 Schubfächer und 280 Liter Volumen",
    "Medikamentenkühlschrank für Apotheken mit präziser Temperaturkontrolle",
    "Gefrierschrank -80°C für langfristige Impfstoff-Lagerung und Biosamples",
    "Standard Haushaltskühlschrank mit Gefrierfach und Energieeffizienz A++",
    "Blutkühlschrank für Krankenhäuser mit Alarmfunktion und Backup-System"
]

search_system.create_index(documents)

# Test searches
test_queries = [
    "Kühlschrank für Medikamente",
    "HMFvh 4001",
    "Krankenhaus Kühlung",
    "Energiesparend"
]

for query in test_queries:
    print(f"\n=== Search: '{query}' ===")

    # Compare different modes
    semantic_results = search_system.search(query, top_k=3, mode="semantic")
    print("Semantic Results:")
    for i, result in enumerate(semantic_results, 1):
        print(f"  {i}. {result['score']:.3f} - {result['document'][:60]}...")
```

## 🛠️ Tools & Frameworks

### **Search Engines**
- **Elasticsearch** - Enterprise-grade full-text search
- **OpenSearch** - Open-source Elasticsearch alternative
- **Solr** - Apache Lucene-based search platform
- **Typesense** - Fast, typo-tolerant search engine

### **Vector Search**
- **Qdrant** - High-performance vector database
- **Pinecone** - Managed vector database service
- **ChromaDB** - Simple, developer-friendly vector DB
- **Weaviate** - GraphQL vector search engine

### **Libraries & SDKs**
- **sentence-transformers** - Semantic embeddings
- **rank-bm25** - BM25 implementation
- **haystack** - Production NLP pipelines
- **langchain** - LLM application framework

### **Embedding Models**
- **multilingual-e5-large** - Best multilingual performance
- **bge-large-en** - State-of-the-art English embeddings
- **instructor-xl** - Task-specific embeddings
- **OpenAI text-embedding-3** - Commercial API option

## 🚀 Was du danach kannst

**Grundlagen:**
- Du verstehst verschiedene Search-Paradigmen (Keyword, Semantic, Hybrid)
- Du implementierst BM25 und Dense Embedding Search von Grund auf
- Du bewertest Search-Qualität mit Precision@K und MRR

**Production-Skills:**
- Du entwickelst skalierbare Search-Systeme mit Vector Databases
- Du optimierst Hybrid Search mit verschiedenen Fusion-Strategien
- Du implementierst Query Analysis und Result Re-ranking

**Advanced:**
- Du kennst moderne Search-Techniken (Neural Retrieval, ColBERT)
- Du integrierst Search in komplexere AI-Systeme (RAG, Agents)
- Du verstehst Performance-Trade-offs zwischen verschiedenen Ansätzen

## 🔗 Weiterführende Themen
- **RAG Systems**: [01-rag-systems.md](01-rag-systems.md) für search-basierte Wissenssysteme
- **Embeddings**: [../03-core/embeddings/](../03-core/embeddings/) für tieferes Embedding-Verständnis
- **Evaluation**: [../03-core/evaluation/03-ranking-metrics.md](../03-core/evaluation/03-ranking-metrics.md) für Search-Metriken
- **Vector Databases**: [../03-core/infrastructure/](../03-core/infrastructure/) für skalierbare Infrastruktur