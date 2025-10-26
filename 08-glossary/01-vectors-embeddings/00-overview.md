# Vectors & Embeddings: Glossar

## 🎯 Übersicht

Diese Kategorie definiert **fundamentale Konzepte der Vektorrepräsentation** - das mathematische Fundament für semantische Suche, RAG-Systeme und moderne NLP.

**Kernfrage:** Wie werden Texte zu Vektoren und wie vergleicht man sie?

---

## 📋 Begriffe in dieser Kategorie

### **Grundlagen (2 Begriffe)**
1. **[01-embedding.md](01-embedding.md)** - Embedding / Dense Vector / Embedding Vector
   - Was ist ein Embedding?
   - Mathematische Definition
   - Warum Embeddings semantische Suche ermöglichen

2. **[04-dot-product.md](04-dot-product.md)** - Dot Product / Inner Product / Skalarprodukt
   - Mathematische Operation
   - Verbindung zur Cosine Similarity
   - Performance-Implikationen

### **Vergleichsoperationen (2 Begriffe)**
3. **[02-cosine-similarity.md](02-cosine-similarity.md)** - Cosine Similarity / Kosinusähnlichkeit
   - **KRITISCH**: Meistgenutzte Ähnlichkeitsmetrik
   - Mathematische Ableitung
   - Warum "Winkel" statt "Distanz"?

4. **[03-vector-normalization.md](03-vector-normalization.md)** - Vector Normalization / L2-Norm
   - Warum Vektoren normalisieren?
   - Verbindung zu Cosine Similarity
   - Performance vs. Korrektheit

### **Retrieval-Methoden (2 Begriffe)**
5. **[05-dense-retrieval.md](05-dense-retrieval.md)** - Dense Retrieval / Neural Search
   - **KRITISCH**: Basis für moderne RAG-Systeme
   - Embedding-basierte Suche
   - Wann besser als Sparse?

6. **[06-sparse-retrieval.md](06-sparse-retrieval.md)** - Sparse Retrieval / BM25 / Lexical Search
   - Traditionelle keyword-basierte Suche
   - BM25 Algorithmus
   - Wann besser als Dense?

---

## 🔗 Lernpfad: Empfohlene Reihenfolge

```
1. Embedding (01) → verstehe was Vektoren sind
   ↓
2. Dot Product (04) → mathematische Grundoperation
   ↓
3. Vector Normalization (03) → Vektoren vergleichbar machen
   ↓
4. Cosine Similarity (02) → Ähnlichkeit messen
   ↓
5. Dense Retrieval (05) → Embeddings für Suche nutzen
   ↓
6. Sparse Retrieval (06) → Alternative Methode verstehen
```

---

## 🎓 Was du danach kannst

Nach Durcharbeiten dieser 6 Begriffe kannst du:

- ✅ **Erklären** warum Embeddings semantische Suche ermöglichen
- ✅ **Berechnen** Cosine Similarity zwischen zwei Vektoren (von Hand!)
- ✅ **Entscheiden** wann Dense vs. Sparse Retrieval besser ist
- ✅ **Implementieren** einen einfachen Dense Retrieval Prototyp
- ✅ **Verstehen** warum Normalisierung wichtig für Performance ist

---

## 🔗 Verwandte Themen im Kompendium

### **Tiefe Theorie:**
- `03-core/02-embeddings/01-vector-fundamentals.md` - Vollständige mathematische Grundlagen
- `03-core/02-embeddings/02-embedding-architectures.md` - Wie Embeddings trainiert werden

### **Praktische Anwendung:**
- `06-applications/01-rag-systems.md` - Embeddings in RAG-Architekturen
- `06-applications/02-search-systems.md` - Dense vs. Sparse im Search-Kontext

### **Advanced Topics:**
- `04-advanced/01-retrieval-methods.md` - Hybrid Retrieval (Dense + Sparse)
- `04-advanced/02-retrieval-optimization.md` - Re-Ranking, Query Transformation

### **Evaluation:**
- `03-core/03-evaluation/01-data-metrics/02-similarity-measures.md` - Weitere Ähnlichkeitsmetriken
- `03-core/03-evaluation/02-ai-evaluation/03-retrieval-metrics.md` - Retrieval-Metriken (Precision@K, NDCG)

---

## 📊 Cross-Reference Matrix

| Begriff | Verwendet in | Voraussetzung für |
|---------|--------------|-------------------|
| **Embedding** | 19 Dateien | Alle Retrieval-Methoden, RAG |
| **Cosine Similarity** | 14 Dateien | Dense Retrieval, Re-Ranking |
| **Vector Normalization** | 8 Dateien | Cosine Similarity Optimierung |
| **Dot Product** | 12 Dateien | Cosine Similarity, Attention |
| **Dense Retrieval** | 15 Dateien | RAG, Hybrid Search |
| **Sparse Retrieval** | 12 Dateien | Hybrid Search, Baseline |

---

**Navigation:**
- 🏠 [Zurück zum Glossar](../00-overview.md)
- ➡️ [Nächste Kategorie: Transformers & Attention](../02-transformers-attention/)
