# Similarity Measures

## 🎯 Lernziele
- Verstehe verschiedene Similarity-Metriken für Embeddings
- Lerne wann welche Metrik zu verwenden ist
- Verstehe die Performance-Implications verschiedener Metriken
- Lerne praktische Implementierung in RAG-Systemen

## 📖 Geschichte & Kontext
Similarity-Measures sind das Herzstück jeder Suchmaschine und RAG-Systems. Die Wahl der richtigen Metrik entscheidet darüber, welche Dokumente als relevant erkannt werden. Von einfachen Keyword-Matching bis zu modernen Embedding-Similarity haben sich diese Metriken stetig weiterentwickelt.

## 🧮 Theorie

## 1. Cosine Similarity ⭐

**Definition:** Misst den Winkel zwischen zwei Vektoren (0° = identisch, 90° = orthogonal)

**Formel:**
```
cos(θ) = (a · b) / (|a| * |b|)
Range: [-1, 1], meist [0, 1] bei Embeddings
```

**Code:**
```python
# sklearn
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([query_emb], embeddings)[0]

# sentence-transformers
from sentence_transformers import util
similarity = util.cos_sim(query_emb, embeddings)[0]

# numpy (wenn normalisiert)
similarity = embeddings @ query_emb  # Dot product = Cosine!
```

**Interpretation:**
- `> 0.8`: Sehr ähnlich
- `0.6 - 0.8`: Ähnlich
- `0.4 - 0.6`: Schwach ähnlich
- `< 0.4`: Unähnlich

**Wann nutzen?**
- ✅ Standard für Semantic Search
- ✅ Unabhängig von Vektor-Länge
- ✅ Development/Testing (explizit, klar)
- ❌ Etwas langsamer als Dot Product

**Target:** Relevante Chunks sollten > 0.6 Similarity haben

---

## 2. Dot Product

**Definition:** Summe der elementweisen Multiplikation zweier Vektoren

**Formel:**
```
a · b = Σ(a[i] * b[i])
Range: [-∞, +∞]
Wenn normalisiert: identisch zu Cosine Similarity
```

**Code:**
```python
# numpy
import numpy as np

# Empfohlen: @ operator
scores = embeddings @ query_emb

# Alternative: np.dot
scores = np.dot(embeddings, query_emb)
```

**Interpretation:**
- Höherer Score = ähnlicher
- Nur zuverlässig wenn Embeddings normalisiert sind
- Dann: identisch zu Cosine (aber schneller!)

**Wann nutzen?**
- ✅ **Production RAG** (schnellste Methode)
- ✅ Wenn Embeddings normalisiert sind
- ✅ Große Datenmengen (Millionen Vektoren)
- ⚠️ Nur mit Normalisierung zuverlässig

**Target:** Normalisierung aktivieren: `model.encode(..., normalize_embeddings=True)`

---

## 3. Euclidean Distance (L2)

**Definition:** Geradlinige Distanz zwischen zwei Punkten im Vektorraum

**Formel:**
```
dist = √(Σ(a[i] - b[i])²)
Range: [0, ∞], 0 = identisch
```

**Code:**
```python
# sklearn
from sklearn.metrics.pairwise import euclidean_distances
distances = euclidean_distances([query_emb], embeddings)[0]

# numpy
distances = np.linalg.norm(embeddings - query_emb, axis=1)

# Zu Similarity konvertieren
similarities = 1 / (1 + distances)
```

**Interpretation:**
- Kleinere Distance = ähnlicher
- Misst absolute Position im Raum
- Geometrisch intuitiv (Luftlinie)

**Wann nutzen?**
- ✅ Clustering (K-Means nutzt Euclidean)
- ✅ Visualisierung (geometrisch intuitiv)
- ✅ Wenn Magnitude wichtig ist
- ❌ Nicht Standard für Semantic Search

**Target:** Nur für spezielle Use Cases, nicht für reguläres Retrieval

---

## 4. Manhattan Distance (L1)

**Definition:** Summe der absoluten Differenzen (Taxicab-Metrik)

**Formel:**
```
dist = Σ|a[i] - b[i]|
Range: [0, ∞]
```

**Code:**
```python
# sklearn
from sklearn.metrics.pairwise import manhattan_distances
distances = manhattan_distances([query_emb], embeddings)[0]

# numpy
distances = np.sum(np.abs(embeddings - query_emb), axis=1)
```

**Interpretation:**
- "Stadtstrecke" statt Luftlinie
- Robust gegen Outlier-Dimensionen
- Kleinere Distance = ähnlicher

**Wann nutzen?**
- ✅ Sparse, hochdimensionale Vektoren
- ✅ Robustness gegen Outliers wichtig
- ❌ Sehr selten in RAG verwendet

**Target:** Nische - nur für spezifische Anforderungen

---

## 📊 Vergleich: Wann welche Similarity?

| Methode | Speed | Normalisierung nötig? | Use Case |
|---------|-------|----------------------|----------|
| **Dot Product** | ⚡⚡⚡ Fastest | Ja (sonst unzuverlässig) | **Production RAG** (Standard) |
| **Cosine Similarity** | ⚡⚡ Fast | Nein | Development/Testing |
| **Euclidean** | ⚡ Slower | Nein | Clustering, Visualisierung |
| **Manhattan** | ⚡ Slower | Nein | Sparse Data, Robustness |

**Empfehlung für RAG:**
```python
# In Production:
scores = embeddings @ query_emb  # Dot product (schnell)

# Voraussetzung:
model.encode(..., normalize_embeddings=True)
```

## 📊 Vergleiche & Varianten

### Performance-Vergleich (1M Embeddings)
- **Dot Product**: ~10ms (GPU), ~50ms (CPU)
- **Cosine Similarity**: ~15ms (GPU), ~80ms (CPU)
- **Euclidean Distance**: ~20ms (GPU), ~120ms (CPU)

### Similarity vs Distance Conversion
```python
# Distance zu Similarity
similarity = 1 / (1 + distance)  # Für Euclidean/Manhattan
similarity = 1 - (distance / max_distance)  # Normalisiert

# Similarity zu Distance
distance = np.sqrt(2 * (1 - similarity))  # Für Cosine
```

### Framework-spezifische Optimierungen
- **FAISS**: Optimierte IP (Inner Product) und L2 Distance
- **ChromaDB**: Standard Cosine, mit IP-Option
- **Weaviate**: Cosine als Standard, Dot als Option

## 🎓 Weiterführende Themen

### Original Papers
- [A Study of Cross-Encoder and Bi-Encoder Architectures](https://arxiv.org/abs/1908.10084) - Reimers & Gurevych
- [Efficient Vector Similarity Computation](https://arxiv.org/abs/1909.12095) - FAISS Technical Paper

### Verwandte Kapitel
- [../embeddings/01-FUNDAMENTALS.md](../../05-embeddings/01-FUNDAMENTALS.md) - Embedding-Grundlagen
- [03-ranking-metrics.md](03-ranking-metrics.md) - Ranking-Evaluation für RAG
- [../infrastructure/03-vector-databases.md](../infrastructure/03-vector-databases.md) - Production Similarity

### Nächste Schritte im Lernpfad
1. **Für RAG-Entwicklung**: [03-ranking-metrics.md](03-ranking-metrics.md)
2. **Für Production**: [../infrastructure/03-vector-databases.md](../infrastructure/03-vector-databases.md)
3. **Für Advanced Retrieval**: [../../advanced/01-retrieval-methods.md](../../advanced/01-retrieval-methods.md)

## 📚 Ressourcen

### Wissenschaftliche Papers
- [Vector Similarity Search](https://arxiv.org/abs/1909.12095) - FAISS Documentation
- [Sentence-BERT Evaluation](https://arxiv.org/abs/1908.10084) - Similarity Benchmarks

### Blog Posts & Tutorials
- [Cosine vs Dot Product](https://blog.reachsumit.com/posts/2019/cosine-vs-dot/) - Detailed Comparison
- [FAISS Index Guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) - Production Similarity

### Videos & Talks
- [Vector Search Explained](https://www.youtube.com/watch?v=QvKMwLjdK-s) - Weaviate Tutorial

### Interaktive Demos
- [Vector Similarity Calculator](https://colab.research.google.com/drive/1eGzCrY1dE_fZ7VhY8H-v-n3K9s_xmwuN) - Try Different Metrics
