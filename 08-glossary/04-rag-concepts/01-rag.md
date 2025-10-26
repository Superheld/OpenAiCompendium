# RAG / Retrieval-Augmented Generation

## Quick Definition

Kombination von **Retrieval** (Suche relevanter Dokumente) + **Generation** (LLM erstellt Antwort basierend auf Retrieved Context) - löst Hallucination-Problem durch externe Knowledge Base.

**Kategorie:** RAG Concepts
**Schwierigkeit:** Intermediate
**Aliases:** RAG, Retrieval-Augmented Generation, Grounded Generation

---

## 🧠 Detaillierte Erklärung

### Das Problem das RAG löst

**LLM-Only (ohne RAG):**
```
User: "Ist der Laborkühlschrank defekt?"
LLM: "Ich habe keine Informationen über spezifische Kühlschränke."
```
→ LLM kennt nur Training-Data, keine aktuellen/spezifischen Daten ❌

**RAG (mit External Knowledge):**
```
1. RETRIEVE: Suche "Laborkühlschrank Status" in Company-DB
   → Findet: "Laborkühlschrank LK-42 defekt seit 2025-01-15"

2. AUGMENT: Füge Retrieved Info zum Prompt hinzu
   Context: "Laborkühlschrank LK-42 defekt seit 2025-01-15"
   Query: "Ist der Laborkühlschrank defekt?"

3. GENERATE: LLM antwortet basierend auf Context
   LLM: "Ja, der Laborkühlschrank LK-42 ist seit 15.01.2025 defekt."
```
→ Faktisch korrekt, grounded in Company-Data ✅

### RAG Pipeline (3 Schritte)

$$\text{RAG}(\text{query}) = \text{Generate}(\text{query}, \text{Retrieve}(\text{query}, \text{Knowledge Base}))$$

**1. Indexing** (Offline, einmalig)
```python
docs = load_documents()  # Handbücher, Wikis, DBs
chunks = chunk_documents(docs, chunk_size=512)
embeddings = embed(chunks)  # Text → Vectors
vector_db.store(embeddings, chunks)  # In Vector DB speichern
```

**2. Retrieval** (Query-Time)
```python
query_embedding = embed(query)
top_k_chunks = vector_db.search(query_embedding, k=5)  # Top-5 ähnlichste
```

**3. Generation** (Query-Time)
```python
context = "\n".join(top_k_chunks)
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
response = llm.generate(prompt)
```

---

## 💻 Code-Beispiel (Minimal RAG)

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========================================
# MINIMAL RAG IMPLEMENTATION
# ========================================

# 1. INDEXING (Offline)
documents = [
    "Laborkühlschrank LK-42 defekt seit 15.01.2025",
    "Serverrack überhitzt, Wartung geplant",
    "Kaffeemaschine wurde repariert",
]

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# 2. RETRIEVAL (Query-Time)
query = "Ist der Kühlschrank kaputt?"
query_embedding = model.encode([query], normalize_embeddings=True)

# Cosine Similarity
scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# Top-K
k = 2
top_k_indices = np.argsort(scores)[::-1][:k]
retrieved_docs = [documents[i] for i in top_k_indices]

print(f"Query: {query}\n")
print("Retrieved Context:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"  {i}. {doc} (score: {scores[top_k_indices[i-1]]:.3f})")

# 3. GENERATION (würde echtes LLM nutzen)
context = "\n".join(retrieved_docs)
prompt = f"""Context:
{context}

Question: {query}
Answer based on the context above:"""

print(f"\n🤖 LLM Prompt:\n{prompt}")
# response = llm.generate(prompt)  # Mit echtem LLM
```

**Output:**
```
Query: Ist der Kühlschrank kaputt?

Retrieved Context:
  1. Laborkühlschrank LK-42 defekt seit 15.01.2025 (score: 0.672)
  2. Kaffeemaschine wurde repariert (score: 0.234)

🤖 LLM Prompt:
Context:
Laborkühlschrank LK-42 defekt seit 15.01.2025
Kaffeemaschine wurde repariert

Question: Ist der Kühlschrank kaputt?
Answer based on the context above:
```

---

## 🔗 Related Terms

### **Komponenten**
- **[Dense Retrieval](../01-vectors-embeddings/05-dense-retrieval.md)**: Embedding-basierte Suche
- **[Chunking](02-chunking.md)**: Dokumente in Chunks splitten
- **[Context Window](../02-transformers-attention/05-context-window.md)**: Wie viele Chunks passen

### **Qualität**
- **[Hallucination](../05-llm-training/04-hallucination.md)**: RAG reduziert Hallucinations
- **Faithfulness**: Antwort grounded in Retrieved Context

---

## 📍 Where This Appears

### **Primary Chapter**
- `06-applications/01-rag-systems.md` - Vollständige RAG-Architekturen
- `04-advanced/01-retrieval-methods.md` - Advanced Retrieval

---

## ⚠️ Common Mistakes

❌ **Retrieval-Only** → Kein LLM, nur Suche (nicht RAG!)
❌ **Generation-Only** → Kein Retrieval (halluziniert!)
❌ **Poor Chunking** → Irrelevante Chunks retrieved
❌ **No Re-Ranking** → Top-K nicht optimal
❌ **Ignoring Faithfulness** → LLM ignoriert Context

---

## 🎯 Zusammenfassung

**RAG in 3 Schritten:**
1. **Retrieve** relevante Dokumente (Dense Search)
2. **Augment** LLM-Prompt mit Retrieved Context
3. **Generate** Antwort grounded in Context

**Vorteile:**
- ✅ Reduziert Hallucinations
- ✅ Aktuelle/spezifische Daten (nicht nur Training-Data)
- ✅ Transparent (zeige Retrieved Sources)

**Trade-off:**
- Retrieval-Qualität bestimmt Output-Qualität
- Latenz höher (Retrieval + Generation)

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ➡️ [Nächster Begriff: Chunking](02-chunking.md)
