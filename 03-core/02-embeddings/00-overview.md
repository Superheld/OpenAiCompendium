# Embeddings: Von Vektoren zu Semantischem Verständnis

## 🎯 Ziel

Beherrsche Embeddings von mathematischen Grundlagen bis zur Production - verstehe Architekturen, wähle Models strategisch, skaliere auf Millionen Dokumente.

## 📖 Geschichte & Kontext

Embeddings verwandeln Text in Vektoren - das Fundament ALLER modernen AI-Systeme. Ohne Embeddings: Kein ChatGPT, kein Google Search, kein Recommendation System.

**Meilensteine:**
- **2013**: Word2Vec - Erste praktische Embeddings
- **2017**: Transformer - "Attention is All You Need"
- **2018**: BERT - Kontextualisierte Embeddings
- **2019**: Sentence-BERT - Optimiert für Similarity
- **2024**: Multimodale Embeddings, SOTA Models

**Warum wichtig:**
```python
# Computer versteht kein Text:
"Laborkühlschrank" → ???

# Aber Vektoren:
[0.234, -0.456, 0.123, ..., -0.234] → Rechnen möglich!
```

## 📂 Kapitel in diesem Abschnitt

```
03-core/02-embeddings/
├── 00-overview.md                   (Diese Datei)
├── 01-vector-fundamentals.md        (Vektorräume, Similarity, Grundlagen)
├── 02-embedding-architectures.md    (Dense, Sparse, Multi-Vector, Cross-Encoder)
├── 03-model-selection.md            (Welches Model? Embedding Spaces verstehen!)
└── 04-vector-databases.md           (Vector DBs, Quantization, Scale)
```

### **[01-vector-fundamentals.md](01-vector-fundamentals.md)** - Vector Fundamentals
**Das Problem:** Warum Computer Text nicht verstehen können

**Was du lernst:**
- Vektorräume: Von Text zu Zahlen
- Distanzmetriken: Cosine, Euclidean, Dot Product
- Ähnlichkeit vs. Relevanz (kritisch!)
- Normalisierung und Visualisierung (t-SNE, UMAP)

**Hands-On:** Similarity-Berechnungen from scratch, Embedding-Space visualisieren

**Warum wichtig:** Ohne Vektor-Verständnis keine Embeddings

---

### **[02-embedding-architectures.md](02-embedding-architectures.md)** - Embedding Architectures
**Das Problem:** Dense findet Synonyme, Sparse findet Keywords - brauchst du beides?

**Was du lernst:**
- **Dense** (Sentence-BERT): Semantisches Verständnis
- **Sparse** (BM25, SPLADE): Exakte Keyword-Matches
- **Multi-Vector** (ColBERT): Token-Level Präzision
- **Cross-Encoder**: Maximale Genauigkeit für Re-Ranking

**Trade-offs:**
| Architektur | Speed | Qualität | Speicher | Use Case |
|-------------|-------|----------|----------|----------|
| Dense | ⚡⚡⚡ | ⭐⭐⭐ | Mittel | Standard |
| Sparse | ⚡⚡⚡ | ⭐⭐ | Gering | Keywords, IDs |
| Multi-Vector | ⚡⚡ | ⭐⭐⭐⭐⭐ | Hoch | High-Precision |
| Cross-Encoder | ⚡ | ⭐⭐⭐⭐⭐ | Mittel | Re-Ranking only |

**Hands-On:** Alle 4 Architekturen im direkten Vergleich

**Warum wichtig:** Architektur-Wahl bestimmt Performance

---

### **[03-model-selection.md](03-model-selection.md)** - Model Selection
**Das Problem:** Embedding Spaces sind Model-spezifisch - Mix = Disaster!

**Was du lernst:**
- **Embedding Spaces**: Warum Models inkompatibel sind
- **Model-Auswahl**: Sprache, Domain, Qualität vs. Speed
- **MTEB Benchmark**: Richtig interpretieren
- **Evaluation**: Model-abhängig, nicht absolut

**Kritisches Konzept:**
```python
# ❌ FALSCH: Models mischen
vectordb.add(docs[:500], model_a.encode(docs[:500]))
vectordb.add(docs[500:], model_b.encode(docs[500:]))
# → Retrieval komplett kaputt!

# ✓ RICHTIG: Ein Model pro DB
vectordb.add(docs, model_a.encode(docs))
```

**Model-Empfehlungen:**
- **Prototyping (Deutsch)**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Production (Multilingual)**: `intfloat/multilingual-e5-large`
- **Production (Englisch)**: `all-mpnet-base-v2`

**Hands-On:** Benchmark mehrerer Models auf eigenem Dataset

**Warum wichtig:** Falsches Model = 40%+ Qualitätsverlust

---

### **[04-vector-databases.md](04-vector-databases.md)** - Vector Databases & Production
**Das Problem:** Von Prototyping (1k Docs) zu Production (10M+ Docs)

**Was du lernst:**
- **Vector Databases**: ChromaDB, Qdrant, Pinecone - wann was?
- **ANN-Algorithmen**: HNSW, IVF - O(log n) statt O(n)
- **Quantization**: Int8, Binary (75-96% Speicher-Ersparnis)
- **Deployment**: Microservices, Caching, Blue-Green

**Speicher-Optimierung:**
```python
# Float32: 1M Docs × 768 dims × 4 bytes = 3 GB
# Int8:    1M Docs × 768 dims × 1 byte  = 768 MB (75% Ersparnis!)
# Binary:  1M Docs × 768 dims × 0.125 B = 96 MB  (96% Ersparnis!)
```

**Vector DB Comparison:**
| DB | Speed | Max Scale | Ease | Cloud |
|----|-------|-----------|------|-------|
| **ChromaDB** | ⚡⚡ | 1M | ⭐⭐⭐ | Nein |
| **Qdrant** | ⚡⚡⚡ | 100M+ | ⭐⭐ | Ja |
| **Pinecone** | ⚡⚡⚡ | Unbegrenzt | ⭐⭐⭐ | Managed |

**Hands-On:** Production-Setup mit Qdrant + Int8 Quantization

**Warum wichtig:** Production ≠ Prototyping - Skalierung entscheidet

---

## 🎯 Lernpfade

### **Path 1: Embedding Fundamentals** (Du willst verstehen wie Embeddings funktionieren)
```
1. 01-vector-fundamentals.md    → Vektorräume & Similarity verstehen
2. 02-embedding-architectures.md → Dense, Sparse, Multi-Vector, Cross-Encoder
3. 03-model-selection.md         → Embedding Spaces & Model-Auswahl
→ Weiter: ../training/contrastive-learning.md (Wie Models trainiert werden)
```

### **Path 2: RAG Builder** (Du willst RAG-Systeme bauen)
```
1. 01-vector-fundamentals.md    → Basics verstehen
2. 02-embedding-architectures.md → Dense + Cross-Encoder lernen
3. 03-model-selection.md         → Richtiges Model wählen
4. 04-vector-databases.md        → Vector DB Setup
→ Weiter: ../../04-advanced/02-retrieval-optimization.md (Chunking, Re-Ranking)
→ Dann: ../../06-applications/rag-systems/ (Praktische Implementierung)
```

### **Path 3: Production Engineer** (Du deployest Embedding-Systeme)
```
1. 03-model-selection.md         → Model-Auswahl strategisch
2. 04-vector-databases.md        → Vector DB + Quantization + Deployment
→ Weiter: ../infrastructure/distributed-systems.md (Horizontal Scaling)
```

## 🛠️ Tools & Frameworks

### Embedding Models
- **Sentence-Transformers** - Standard Library, 100+ Models
- **Hugging Face** - Model Hub
- **OpenAI Embeddings** - text-embedding-3-small/large
- **Cohere** - Multilingual Embeddings API

### Vector Databases
- **ChromaDB** - Einfach, lokal, ideal für Start
- **Qdrant** - Production, Rust, sehr schnell
- **Pinecone** - Managed Cloud, skaliert unbegrenzt
- **Weaviate** - GraphQL, hybrid capabilities
- **Milvus** - Open Source, horizontale Skalierung

## 🚀 Was du danach kannst

**Fundamentales Verständnis:**
- ✓ Du verstehst Vektorräume als mathematisches Konzept
- ✓ Du erkennst Trade-offs zwischen Architekturen (Dense vs. Sparse vs. Multi-Vector)
- ✓ Du verstehst dass Embedding Spaces Model-spezifisch sind

**Architektur-Expertise:**
- ✓ Du wählst zwischen Dense, Sparse, Multi-Vector, Cross-Encoder basierend auf Use Case
- ✓ Du implementierst alle 4 Architekturen praktisch
- ✓ Du verstehst Contrastive Learning und wie Models trainiert werden

**Model-Selektion:**
- ✓ Du wählst richtiges Model für Sprache, Domain, Qualitäts-Anforderungen
- ✓ Du benchmarkst Models auf eigenem Dataset (nicht nur MTEB)
- ✓ Du verstehst dass Model-Wechsel komplette Re-Embedding erfordert

**Production:**
- ✓ Du deployest Vector Database mit Millionen Docs
- ✓ Du nutzt Quantization (int8) für 75% Speicher-Ersparnis
- ✓ Du skalierst Embedding-Systeme mit ANN-Algorithmen (HNSW)
- ✓ Du implementierst Blue-Green Deployment für Model-Updates

## 🔗 Weiterführende Themen

**Retrieval & Optimization:**
→ [../../04-advanced/02-retrieval-optimization.md](../../04-advanced/02-retrieval-optimization.md) - Chunking, Re-Ranking, Hybrid Search

**Praktische Anwendung:**
→ [../../06-applications/rag-systems/](../../06-applications/rag-systems/) - RAG End-to-End implementieren

**Vertiefung:**
→ [../training/contrastive-learning.md](../training/contrastive-learning.md) - Wie Dense Models trainiert werden
→ [../infrastructure/vector-databases.md](../infrastructure/vector-databases.md) - Deep Dive HNSW, IVF

**Evaluation:**
→ [../evaluation/metrics.md](../evaluation/metrics.md) - Recall@k, MRR, nDCG
