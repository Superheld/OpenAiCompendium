# Vector Normalization / L2-Normalisierung

## Quick Definition

Skalierung eines Vektors auf Länge 1, sodass nur die **Richtung** erhalten bleibt, nicht die **Magnitude** (Länge).

**Kategorie:** Vectors & Embeddings
**Schwierigkeit:** Beginner
**Aliases:** Vector Normalization, L2-Norm, Unit Vector, Normalisierung

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

Stell dir einen Pfeil vor:
- **Vor Normalisierung**: Pfeil kann beliebige Länge haben (1m, 5m, 100m)
- **Nach Normalisierung**: Pfeil hat immer Länge = 1 (aber gleiche Richtung!)

**Warum?** Wenn du nur die **Richtung** vergleichen willst (nicht die Länge), normalisiere zuerst.

**Beispiel:**
```
Vektor A: [3, 4]          Länge: 5
Normalisiert: [0.6, 0.8]  Länge: 1 (Richtung identisch!)
```

### Mathematische Formalisierung

Für einen Vektor $\mathbf{v} = [v_1, v_2, \ldots, v_n]$:

**L2-Norm (Vektorlänge):**
$$||\mathbf{v}||_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

**Normalisierter Vektor:**
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{||\mathbf{v}||_2} = \left[\frac{v_1}{||\mathbf{v}||_2}, \frac{v_2}{||\mathbf{v}||_2}, \ldots, \frac{v_n}{||\mathbf{v}||_2}\right]$$

**Eigenschaft:**
$$||\hat{\mathbf{v}}||_2 = 1 \quad \text{(Länge ist immer 1)}$$

**Beispiel-Berechnung:**
```
v = [3, 4]
||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5

v̂ = [3/5, 4/5] = [0.6, 0.8]
||v̂|| = √(0.6² + 0.8²) = √(0.36 + 0.64) = √1 = 1 ✓
```

### Why It Matters

**1. Cosine Similarity = Dot Product** (wenn normalisiert)

**Ohne Normalisierung:**
$$\text{cosine}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||} \quad \text{(aufwendig!)}$$

**Mit Normalisierung** ($||\mathbf{A}|| = ||\mathbf{B}|| = 1$):
$$\text{cosine}(\mathbf{A}, \mathbf{B}) = \mathbf{A} \cdot \mathbf{B} \quad \text{(nur Dot Product!)}$$

**Performance-Gewinn:** 10× schneller in Vector Databases!

**2. Fair Comparison**

Ohne Normalisierung bevorzugen Distanz-Metriken kürzere Vektoren:

```python
# Nicht-normalisiert
doc1_emb = [0.1, 0.2, 0.1]  # ||v|| = 0.24
doc2_emb = [1.0, 2.0, 1.0]  # ||v|| = 2.45 (10× länger!)

euclidean_distance(doc1_emb, doc2_emb)  # Groß, wegen Längen-Unterschied
```

**Mit Normalisierung:**
```python
doc1_norm = normalize(doc1_emb)  # ||v|| = 1
doc2_norm = normalize(doc2_emb)  # ||v|| = 1

cosine_similarity(doc1_norm, doc2_norm)  # Fair, nur Richtung zählt!
```

**3. Standard in Embedding-Modellen**

Viele Embedding-Modelle normalisieren bereits automatisch:
- **Sentence-BERT**: `normalize_embeddings=True` (default)
- **OpenAI ada-002**: Pre-normalized
- **E5, BGE**: Normalisiert während Training

**Check ob normalisiert:**
```python
embedding = model.encode("Text")
print(np.linalg.norm(embedding))  # Sollte ≈ 1.0 sein
```

### Common Variations

**1. L2-Normalisierung** (Standard)
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{||\mathbf{v}||_2} = \frac{\mathbf{v}}{\sqrt{\sum v_i^2}}$$

**2. L1-Normalisierung** (selten für Embeddings)
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{||\mathbf{v}||_1} = \frac{\mathbf{v}}{\sum |v_i|}$$

**3. Max-Normalisierung** (für Bilder)
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\max(|v_i|)}$$

**Für Embeddings:** Immer L2-Norm!

---

## 💻 Code-Beispiel

```python
import numpy as np

# Original Embedding (nicht normalisiert)
embedding = np.array([3.0, 4.0, 0.0])
print(f"Original: {embedding}")
print(f"Länge: {np.linalg.norm(embedding):.3f}")  # 5.0

# Methode 1: NumPy (manuell)
norm = np.linalg.norm(embedding)
normalized = embedding / norm
print(f"\nNormalisiert: {normalized}")
print(f"Länge: {np.linalg.norm(normalized):.3f}")  # 1.0

# Methode 2: sklearn (batch-fähig)
from sklearn.preprocessing import normalize

embeddings = np.array([
    [3.0, 4.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0]
])
normalized_batch = normalize(embeddings, norm='l2', axis=1)
print(f"\nBatch normalisiert:\n{normalized_batch}")
print(f"Längen: {np.linalg.norm(normalized_batch, axis=1)}")  # [1. 1. 1.]

# Anwendung: Cosine Similarity via Dot Product
emb1_norm = normalize(embeddings[[0]])[0]
emb2_norm = normalize(embeddings[[1]])[0]
similarity = np.dot(emb1_norm, emb2_norm)
print(f"\nCosine Similarity (via Dot Product): {similarity:.3f}")
```

**Output:**
```
Original: [3. 4. 0.]
Länge: 5.000

Normalisiert: [0.6 0.8 0. ]
Länge: 1.000

Batch normalisiert:
[[0.6        0.8        0.        ]
 [1.         0.         0.        ]
 [0.         0.70710678 0.70710678]]
Längen: [1. 1. 1.]

Cosine Similarity (via Dot Product): 0.600
```

---

## 🔗 Related Terms

### **Voraussetzungen**
- **[Embedding](01-embedding.md)**: Was normalisiert wird
- **Vektor-Länge / Magnitude**: $||\mathbf{v}||$

### **Nutzt Normalisierung**
- **[Cosine Similarity](02-cosine-similarity.md)**: Wird trivial mit normalisierten Vektoren
- **[Dense Retrieval](05-dense-retrieval.md)**: Vector DBs nutzen normalisierte Embeddings

### **Verwandt**
- **[Dot Product](04-dot-product.md)**: Wird zu Cosine Similarity bei normalisierten Vektoren

---

## 📍 Where This Appears

### **Primary Chapter**
- `03-core/02-embeddings/01-vector-fundamentals.md` (Sektion 2.3) - Mathematische Grundlagen
- `03-core/03-evaluation/01-data-metrics/02-similarity-measures.md` - Normalisierung für Metriken

### **Implementation**
- `03-core/05-infrastructure/02-model-serving.md` (Sektion 3) - Vector DB Optimierung
- `03-core/02-embeddings/04-vector-databases.md` - Normalisierung in Pinecone, Weaviate

### **Usage Examples**
- `06-applications/01-rag-systems.md` (Sektion 2.2) - RAG Embeddings normalisieren
- `04-advanced/02-retrieval-optimization.md` - Normalisierung für Re-Ranking

---

## ⚠️ Common Misconceptions

### ❌ "Normalisierung verändert semantische Bedeutung"
**Falsch!** Nur die **Länge** ändert sich, nicht die **Richtung** (Semantik).

**Beweis:**
```python
original = [3, 4]
normalized = [0.6, 0.8]

# Gleiche Richtung!
angle_original = np.arctan2(4, 3)      # 0.927 rad
angle_normalized = np.arctan2(0.8, 0.6)  # 0.927 rad (identisch!)
```

**Richtig:** Semantische Ähnlichkeit bleibt erhalten, nur Längen-Information geht verloren.

### ❌ "Alle Embedding-Modelle geben normalisierte Vektoren zurück"
**Falsch!** Manche Modelle normalisieren, manche nicht.

**Check:**
```python
embedding = model.encode("Text")
length = np.linalg.norm(embedding)
print(f"Länge: {length}")

# Normalisiert: length ≈ 1.0
# Nicht normalisiert: length > 1.0
```

**Best Practice:** Immer selbst normalisieren, außer du bist sicher:
```python
from sklearn.preprocessing import normalize
embeddings = normalize(embeddings, norm='l2')
```

### ❌ "Normalisierung ist optional"
**Kommt drauf an!**

**MUSS normalisieren:**
- Vector DBs die Dot Product nutzen (z.B. FAISS mit InnerProduct)
- Wenn Cosine Similarity via Dot Product berechnet werden soll
- Bei unterschiedlich langen Dokumenten

**KANN normalisieren:**
- Wenn Vector DB Cosine Similarity direkt unterstützt
- sklearn `cosine_similarity()` macht es automatisch

**Best Practice:** Normalisiere beim **Einfügen in Vector DB** (einmalig), nicht bei jeder Query!

---

## 🎯 Zusammenfassung

**Ein Satz:** Normalisierung skaliert Vektoren auf Länge 1, sodass nur Richtung (Semantik) zählt, nicht Magnitude.

**Formel (Merksatz):**
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{||\mathbf{v}||}$$

**Key Takeaways:**
1. **Performance:** Cosine = Dot Product (10× schneller!)
2. **Fairness:** Längen-unabhängiger Vergleich
3. **Einmalig:** Normalisiere beim Einfügen, nicht bei jeder Query
4. **Überprüfen:** Nicht alle Modelle normalisieren automatisch

**Wann nutzen?**
- ✅ Vector DBs mit Dot Product Index
- ✅ Cosine Similarity Optimierung
- ✅ Fair Comparison unterschiedlich langer Texte
- ❌ Wenn absolute Vektorlänge Information enthält

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ⬅️ [Vorheriger Begriff: Cosine Similarity](02-cosine-similarity.md)
- ➡️ [Nächster Begriff: Dot Product](04-dot-product.md)
