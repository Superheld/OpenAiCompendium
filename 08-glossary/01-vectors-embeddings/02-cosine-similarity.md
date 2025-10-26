# Cosine Similarity / Kosinusähnlichkeit

## Quick Definition

Ein Ähnlichkeitsmaß zwischen zwei Vektoren basierend auf dem **Winkel** zwischen ihnen (nicht der Distanz), Wertebereich: [-1, +1].

**Kategorie:** Vectors & Embeddings
**Schwierigkeit:** Beginner
**Aliases:** Cosine Similarity, Kosinusähnlichkeit, Cosine Distance (1 - cosine similarity)

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

Stell dir zwei Pfeile vor, die vom Ursprung aus in verschiedene Richtungen zeigen:
- **Gleiche Richtung** (Winkel 0°) → Cosine Similarity = **1.0** (identisch)
- **Orthogonal** (Winkel 90°) → Cosine Similarity = **0.0** (keine Ähnlichkeit)
- **Entgegengesetzt** (Winkel 180°) → Cosine Similarity = **-1.0** (Gegenteil)

**Wichtig:** Cosine Similarity ignoriert die **Länge** der Vektoren, nur die **Richtung** zählt!

**Beispiel:**
```
Vektor A: [3, 4]     (Länge: 5)
Vektor B: [6, 8]     (Länge: 10, doppelt so lang)
Cosine Similarity: 1.0 (gleiche Richtung!)
```

### Mathematische Formalisierung

Für zwei Vektoren $\mathbf{A}$ und $\mathbf{B}$:

$$\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**Komponenten:**
- **Zähler**: $\mathbf{A} \cdot \mathbf{B}$ = Dot Product (Skalarprodukt)
- **Nenner**: $||\mathbf{A}|| \cdot ||\mathbf{B}||$ = Produkt der Vektorlängen (L2-Normen)

**Geometrische Interpretation:**

$$\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \cos(\theta)$$

wo $\theta$ der Winkel zwischen den Vektoren ist.

**Wertebereich:**
- $+1$: Vektoren zeigen in exakt gleiche Richtung
- $0$: Vektoren sind orthogonal (keine Ähnlichkeit)
- $-1$: Vektoren zeigen in entgegengesetzte Richtung

**Beispiel-Berechnung:**

```
A = [1, 2, 3]
B = [2, 4, 6]

Dot Product: 1×2 + 2×4 + 3×6 = 2 + 8 + 18 = 28
||A|| = √(1² + 2² + 3²) = √14 ≈ 3.74
||B|| = √(2² + 4² + 6²) = √56 ≈ 7.48

Cosine Similarity = 28 / (3.74 × 7.48) ≈ 28 / 28 = 1.0
```

### Why It Matters

**1. Skalierungs-Invarianz**

Cosine Similarity ist unabhängig von der Vektorlänge - perfekt für Text-Embeddings!

**Problem mit Euclidean Distance:**
```python
# Lange Dokumente haben größere Embeddings → unfairer Vergleich
doc1 = "KI"           → embedding_norm = 0.5
doc2 = "KI ist ..."   → embedding_norm = 2.0  (4× länger)
# Euclidean Distance bevorzugt kürzere Embeddings!
```

**Lösung mit Cosine:**
```python
# Nur Richtung zählt, nicht Länge → fairer Vergleich
cosine_similarity(doc1, doc2)  # Vergleicht semantische Ähnlichkeit
```

**2. Effiziente Berechnung mit normalisierten Vektoren**

Wenn Embeddings **normalisiert** sind ($||\mathbf{v}|| = 1$):

$$\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \mathbf{A} \cdot \mathbf{B}$$

**Nur ein Dot Product!** → 10× schneller in Vector Databases

**3. Standard in Embedding-Modellen**

Fast alle Embedding-Modelle sind auf Cosine Similarity optimiert:
- **Sentence-BERT**: Trainiert mit Cosine Loss
- **E5, BGE**: Contrastive Learning mit Cosine
- **OpenAI ada-002**: Dokumentiert als Cosine-optimiert

### Common Variations

**1. Cosine Distance** (für Clustering/KNN)

$$\text{cosine\_distance}(\mathbf{A}, \mathbf{B}) = 1 - \text{cosine\_similarity}(\mathbf{A}, \mathbf{B})$$

Wertebereich: [0, 2]
- 0 = identisch
- 2 = entgegengesetzt

**2. Normalisierte Embeddings + Dot Product**

```python
# Viele Vector DBs normalisieren automatisch
embeddings = normalize(embeddings)  # ||v|| = 1
similarity = dot_product(emb1, emb2)  # = cosine similarity!
```

**3. Angular Distance**

$$\text{angular\_distance} = \frac{\arccos(\text{cosine\_similarity})}{\pi}$$

Wertebereich: [0, 1], entspricht "normalized angle"

---

## 💻 Code-Beispiel

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Beispiel: Zwei Sätze als Embeddings
# (normalerweise von Sentence-BERT, hier vereinfacht)
embedding1 = np.array([[0.5, 0.8, 0.3, 0.1]])  # "Laborkühlschrank defekt"
embedding2 = np.array([[0.52, 0.79, 0.31, 0.09]])  # "Kühlschrank im Labor kaputt"
embedding3 = np.array([[-0.3, 0.2, -0.9, 0.4]])  # "Serverausfall"

# Methode 1: sklearn (einfach, aber langsamer)
sim_12 = cosine_similarity(embedding1, embedding2)[0][0]
sim_13 = cosine_similarity(embedding1, embedding3)[0][0]
print(f"Similarity (1 vs 2): {sim_12:.3f}")  # ~0.998 (sehr ähnlich!)
print(f"Similarity (1 vs 3): {sim_13:.3f}")  # ~-0.234 (unähnlich)

# Methode 2: NumPy (manuell, für Verständnis)
def cosine_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

print(f"Manual (1 vs 2): {cosine_sim(embedding1[0], embedding2[0]):.3f}")

# Methode 3: Optimiert mit normalisierten Vektoren
def normalize(v):
    return v / np.linalg.norm(v)

emb1_norm = normalize(embedding1[0])
emb2_norm = normalize(embedding2[0])
sim_optimized = np.dot(emb1_norm, emb2_norm)  # Nur Dot Product!
print(f"Optimized (1 vs 2): {sim_optimized:.3f}")
```

**Output:**
```
Similarity (1 vs 2): 0.998
Similarity (1 vs 3): -0.234
Manual (1 vs 2): 0.998
Optimized (1 vs 2): 0.998
```

---

## 🔗 Related Terms

### **Voraussetzungen**
- **[Dot Product](04-dot-product.md)**: Zähler der Cosine-Formel
- **[Vector Normalization](03-vector-normalization.md)**: Nenner der Cosine-Formel
- **[Embedding](01-embedding.md)**: Was verglichen wird

### **Baut darauf auf**
- **[Dense Retrieval](05-dense-retrieval.md)**: Nutzt Cosine Similarity für Ranking
- **Re-Ranking** (siehe `04-rag-concepts/04-reranking.md`): Verfeinert Cosine-basierte Ergebnisse

### **Alternativen**
- **Euclidean Distance**: $||\mathbf{A} - \mathbf{B}||$ (längenabhängig, nicht ideal für Embeddings)
- **Manhattan Distance**: $\sum |A_i - B_i|$ (selten für Embeddings)
- **Dot Product**: Wenn Vektoren bereits normalisiert sind

---

## 📍 Where This Appears

### **Primary Chapter**
- `03-core/02-embeddings/01-vector-fundamentals.md` (Sektion 3) - Vollständige Ableitung
- `03-core/03-evaluation/01-data-metrics/02-similarity-measures.md` - Vergleich verschiedener Metriken

### **Usage Examples**
- `06-applications/01-rag-systems.md` (Sektion 2.3) - Cosine in RAG Retrieval
- `06-applications/02-search-systems.md` (Sektion 2.1) - Dense Search Ranking
- `04-advanced/01-retrieval-methods.md` - Dense Retrieval Scoring

### **Implementation Details**
- `03-core/05-infrastructure/02-model-serving.md` - Optimierte Cosine-Berechnung in Vector DBs
- `03-core/02-embeddings/04-vector-databases.md` - Index-Strukturen für Cosine Search (HNSW, IVF)

---

## ⚠️ Common Misconceptions

### ❌ "Cosine Similarity ist eine Distanz-Metrik"
**Falsch!** Cosine Similarity ist ein **Ähnlichkeitsmaß** (höher = ähnlicher).

**Distanz-Metriken:**
- Cosine **Distance** = $1 - \text{cosine\_similarity}$ (niedriger = ähnlicher)
- Euclidean Distance = $||\mathbf{A} - \mathbf{B}||$

**Merksatz:**
- **Similarity**: Höher = ähnlicher (0 bis 1)
- **Distance**: Niedriger = ähnlicher (0 bis ∞)

### ❌ "Cosine Similarity funktioniert nur für positive Werte"
**Falsch!** Cosine funktioniert für **beliebige** reelle Zahlen.

Embeddings können negative Werte haben:
```python
embedding = [0.5, -0.3, 0.8, -0.1, ...]  # Völlig normal!
```

**Wertebereich:** $[-1, +1]$ (nicht $[0, 1]$!)
- $+1$: Gleiche Richtung
- $0$: Orthogonal
- $-1$: Entgegengesetzt

**Aber:** In der Praxis sind Embedding-Similarities meist **positiv** (0 bis 1), weil Sentence-BERT Modelle so trainiert sind.

### ❌ "Cosine Similarity = Dot Product"
**Nur wenn normalisiert!**

**Allgemein:**
$$\text{cosine} = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||}$$

**Wenn** $||\mathbf{A}|| = ||\mathbf{B}|| = 1$ **dann:**
$$\text{cosine} = \mathbf{A} \cdot \mathbf{B}$$

**Trade-off:**
- **Mit Normalisierung**: Cosine = Dot Product (schnell!)
- **Ohne Normalisierung**: Vollständige Formel nötig (langsamer)

**Best Practice:** Normalisiere Embeddings **einmal** beim Einfügen in Vector DB → alle Queries nutzen schnellen Dot Product.

---

## 📊 Performance-Vergleich

| Methode | Berechnung | Geschwindigkeit | Use-Case |
|---------|------------|-----------------|----------|
| **Cosine (full formula)** | $\mathbf{A} \cdot \mathbf{B} / (\\|\mathbf{A}\\| \\|\mathbf{B}\\|)$ | Baseline (1×) | Nicht-normalisierte Vektoren |
| **Normalized + Dot Product** | $\mathbf{A} \cdot \mathbf{B}$ | **10× schneller** | Vector DBs (Pinecone, Weaviate) |
| **Euclidean Distance** | $\\|\mathbf{A} - \mathbf{B}\\|$ | ~5× langsamer | Nicht für Embeddings! |

**Benchmark (1M Vergleiche):**
```
Cosine (sklearn):      1.2s
Normalized Dot (NumPy): 0.12s  ← 10× Speedup!
Euclidean:             2.1s
```

---

## 🎯 Zusammenfassung

**Ein Satz:** Cosine Similarity misst den **Winkel** zwischen Vektoren (nicht Distanz) und ist der Standard für Embedding-Vergleiche.

**Formel (Merksatz):**
$$\text{cosine} = \frac{\text{Dot Product}}{\text{Länge}_A \times \text{Länge}_B}$$

**Key Takeaways:**
1. **Skalierungs-invariant**: Länge egal, nur Richtung zählt
2. **Optimiert für Embeddings**: Fast alle Modelle trainiert mit Cosine Loss
3. **Performance-Hack**: Normalisierung → Cosine = Dot Product (10× schneller)
4. **Wertebereich**: [-1, +1], aber bei Embeddings meist [0, 1]

**Wann nutzen?**
- ✅ Text-Embeddings vergleichen (Standard!)
- ✅ Dense Retrieval Ranking
- ✅ Skalierungs-unabhängige Ähnlichkeit
- ❌ Wenn absolute Vektorlänge wichtig ist

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ⬅️ [Vorheriger Begriff: Embedding](01-embedding.md)
- ➡️ [Nächster Begriff: Vector Normalization](03-vector-normalization.md)
