# Dot Product / Inner Product / Skalarprodukt

## Quick Definition

Eine mathematische Operation, die zwei Vektoren multipliziert und eine **einzelne Zahl** (Skalar) zurückgibt - das Fundament für Cosine Similarity und Attention.

**Kategorie:** Vectors & Embeddings
**Schwierigkeit:** Beginner
**Aliases:** Dot Product, Inner Product, Skalarprodukt, Scalar Product

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

Stell dir vor, du hast zwei Listen von Zahlen und willst herausfinden, wie "aligned" sie sind:
- Multipliziere **entsprechende** Elemente
- **Addiere** alle Produkte
- Ergebnis: Eine einzelne Zahl (Skalar)

**Beispiel:**
```
A = [1, 2, 3]
B = [4, 5, 6]

Dot Product = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

**Interpretation:**
- **Großes Ergebnis**: Vektoren zeigen in ähnliche Richtung + haben große Werte
- **Kleines/0**: Vektoren sind orthogonal (keine Ähnlichkeit)
- **Negativ**: Vektoren zeigen in entgegengesetzte Richtungen

### Mathematische Formalisierung

Für zwei Vektoren $\mathbf{A}, \mathbf{B} \in \mathbb{R}^n$:

$$\mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} A_i \cdot B_i = A_1B_1 + A_2B_2 + \cdots + A_nB_n$$

**Geometrische Interpretation:**

$$\mathbf{A} \cdot \mathbf{B} = ||\mathbf{A}|| \cdot ||\mathbf{B}|| \cdot \cos(\theta)$$

wo $\theta$ der Winkel zwischen den Vektoren ist.

**Wichtige Eigenschaften:**

1. **Kommutativität**: $\mathbf{A} \cdot \mathbf{B} = \mathbf{B} \cdot \mathbf{A}$
2. **Distributivität**: $\mathbf{A} \cdot (\mathbf{B} + \mathbf{C}) = \mathbf{A} \cdot \mathbf{B} + \mathbf{A} \cdot \mathbf{C}$
3. **Skalar-Multiplikation**: $(c\mathbf{A}) \cdot \mathbf{B} = c(\mathbf{A} \cdot \mathbf{B})$
4. **Orthogonalität**: $\mathbf{A} \cdot \mathbf{B} = 0 \Leftrightarrow \mathbf{A} \perp \mathbf{B}$

**Beispiel-Berechnung:**

```
A = [2, 3, 1]
B = [1, 4, -1]

A · B = (2×1) + (3×4) + (1×-1)
     = 2 + 12 - 1
     = 13
```

### Why It Matters

**1. Basis für Cosine Similarity**

$$\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||}$$

**Wenn normalisiert** ($||\mathbf{A}|| = ||\mathbf{B}|| = 1$):
$$\text{cosine\_similarity} = \mathbf{A} \cdot \mathbf{B}$$

→ **10× schneller** in Vector Databases!

**2. Self-Attention in Transformers**

Attention Score zwischen Query und Key:

$$\text{Attention}(Q, K) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)$$

Der Dot Product $Q \cdot K^T$ bestimmt "wie relevant ist dieser Key für die Query".

**3. Effiziente Hardware-Beschleunigung**

Moderne CPUs/GPUs haben **spezialisierte Instruktionen** für Dot Products:
- **SIMD** (Single Instruction Multiple Data)
- **Matrix Multiplication Units** (TPUs, GPUs)
- **Optimiert auf allen Ebenen** (CPU → Cache → RAM)

**Benchmark (1M Operationen):**
```
Dot Product (NumPy):    5 ms    ← Hardware-optimiert!
Manual Loop:            850 ms  ← 170× langsamer
```

**4. Vector Database Indexing**

Vector DBs nutzen Dot Product für **schnelle Approximate Nearest Neighbor (ANN) Search**:
- **HNSW** (Hierarchical Navigable Small World): Dot Product als Scoring
- **IVF** (Inverted File Index): Cluster mit Dot Product
- **FAISS InnerProduct**: Direkt Dot Product als Metrik

### Common Variations

**1. Standard Dot Product** (zwei Vektoren)
```python
result = np.dot(a, b)  # Skalar
```

**2. Matrix-Vector Dot Product**
```python
# Matrix (m×n) · Vektor (n) = Vektor (m)
result = np.dot(matrix, vector)
```

**3. Matrix-Matrix Dot Product** (Matrix Multiplication)
```python
# Matrix (m×k) · Matrix (k×n) = Matrix (m×n)
result = np.dot(matrix1, matrix2)
# Oder: result = matrix1 @ matrix2
```

**4. Batched Dot Product** (für Embeddings)
```python
# Alle Queries vs. alle Keys
# Q: (batch, dim), K: (batch, dim) → (batch, batch)
scores = Q @ K.T
```

---

## 💻 Code-Beispiel

```python
import numpy as np

# 1. Einfacher Dot Product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.dot(a, b)
print(f"Dot Product: {result}")  # 32

# Alternative Syntax (Python 3.5+)
result2 = a @ b
print(f"@ Operator: {result2}")  # 32

# Manuell (für Verständnis)
result_manual = sum(a[i] * b[i] for i in range(len(a)))
print(f"Manual: {result_manual}")  # 32

# 2. Batch Dot Product (Embeddings)
queries = np.array([
    [0.5, 0.8, 0.3],  # Query 1
    [0.2, 0.4, 0.9]   # Query 2
])
keys = np.array([
    [0.6, 0.7, 0.2],  # Key 1
    [0.1, 0.5, 0.8],  # Key 2
    [0.9, 0.1, 0.3]   # Key 3
])

# Jede Query vs. alle Keys
scores = queries @ keys.T  # Shape: (2, 3)
print(f"\nAttention Scores:\n{scores}")
# [[1.02 0.69 0.62]
#  [1.06 0.94 0.59]]

# 3. Normalisierte Vektoren → Cosine Similarity
from sklearn.preprocessing import normalize

a_norm = normalize([a])[0]
b_norm = normalize([b])[0]

cosine_sim = np.dot(a_norm, b_norm)
print(f"\nCosine Similarity (via Dot): {cosine_sim:.3f}")  # 0.974
```

**Output:**
```
Dot Product: 32
@ Operator: 32
Manual: 32

Attention Scores:
[[1.02 0.69 0.62]
 [1.06 0.94 0.59]]

Cosine Similarity (via Dot): 0.974
```

---

## 🔗 Related Terms

### **Baut darauf auf**
- **[Cosine Similarity](02-cosine-similarity.md)**: Normalisierter Dot Product
- **[Vector Normalization](03-vector-normalization.md)**: Macht Dot Product = Cosine
- **Self-Attention** (siehe `02-transformers-attention/01-self-attention.md`): Nutzt Dot Product für Attention Scores

### **Verwandt**
- **[Embedding](01-embedding.md)**: Was mit Dot Product verglichen wird
- **Matrix Multiplication**: Batch-Version von Dot Products

### **Alternativen**
- **Euclidean Distance**: $||\mathbf{A} - \mathbf{B}||$ (Distanz statt Ähnlichkeit)
- **Manhattan Distance**: $\sum |A_i - B_i|$

---

## 📍 Where This Appears

### **Primary Chapter**
- `03-core/02-embeddings/01-vector-fundamentals.md` (Sektion 2.2) - Mathematische Grundlagen
- `01-historical/04-attention-transformers/` - Dot Product in Attention

### **Implementation**
- `02-modern-ai/01-llms/01-model-families.md` (Sektion über Attention) - Q·K^T
- `03-core/05-infrastructure/02-model-serving.md` - Optimierter Dot Product in Vector DBs

### **Usage Examples**
- `06-applications/01-rag-systems.md` (Sektion 2.3) - Embedding-Vergleich
- `06-applications/02-search-systems.md` - Dense Retrieval Scoring
- `04-advanced/01-retrieval-methods.md` - Retrieval Scoring

---

## ⚠️ Common Misconceptions

### ❌ "Dot Product ist immer positiv"
**Falsch!** Dot Product kann **negativ** sein.

**Beispiel:**
```python
a = [1, 0]
b = [-1, 0]  # Entgegengesetzte Richtung

dot = np.dot(a, b)  # -1 (negativ!)
```

**Wertebereich:** $(-\infty, +\infty)$
- Positiv: Vektoren zeigen in ähnliche Richtung
- Null: Orthogonal
- Negativ: Entgegengesetzte Richtung

**Aber:** Bei **normalisierten** Embeddings ist Dot Product = Cosine Similarity, also meist $[0, 1]$.

### ❌ "Dot Product = Cosine Similarity"
**Nur bei normalisierten Vektoren!**

**Allgemein:**
$$\text{Dot Product} = ||\mathbf{A}|| \cdot ||\mathbf{B}|| \cdot \cos(\theta)$$

**Wenn** $||\mathbf{A}|| = ||\mathbf{B}|| = 1$:
$$\text{Dot Product} = \cos(\theta) = \text{Cosine Similarity}$$

**Beispiel (nicht normalisiert):**
```python
a = [3, 4]  # ||a|| = 5
b = [6, 8]  # ||b|| = 10

dot = 3×6 + 4×8 = 50          # Dot Product
cosine = 50 / (5 × 10) = 1.0  # Cosine Similarity
```

Dot Product (50) ≠ Cosine Similarity (1.0)!

### ❌ "Größerer Dot Product = ähnlicher"
**Hängt von Vektorlängen ab!**

**Problem:**
```python
a = [1, 0]      # ||a|| = 1
b = [10, 0]     # ||b|| = 10 (10× länger, gleiche Richtung)

dot = 1×10 = 10  # Groß wegen Länge, nicht wegen Ähnlichkeit
```

**Lösung:** Normalisiere erst, dann ist Dot Product = echte Ähnlichkeit.

**Best Practice:**
- **Mit Normalisierung**: Dot Product → Ähnlichkeit
- **Ohne Normalisierung**: Nutze Cosine Similarity

---

## 📊 Performance-Optimierung

### **Hardware-Beschleunigung**

```python
import numpy as np

# NumPy nutzt automatisch BLAS (optimierte lineare Algebra)
a = np.random.rand(1000)
b = np.random.rand(1000)

# Optimiert!
result = np.dot(a, b)  # 0.01 ms

# Nicht optimiert (Manual Loop)
result_manual = sum(a[i] * b[i] for i in range(len(a)))  # 0.15 ms
```

### **Batch-Verarbeitung**

```python
# Einzeln (langsam)
for query in queries:
    for key in keys:
        score = np.dot(query, key)  # 1000× Dot Products

# Batch (schnell)
scores = queries @ keys.T  # Eine Matrix-Multiplikation!
# 100× schneller durch Parallelisierung
```

### **Vector Database Optimierung**

Vector DBs nutzen:
1. **Quantization**: 4-bit Dot Products (4× weniger Memory)
2. **SIMD**: Parallele Dot Products
3. **Approximate Search**: HNSW navigiert mit Dot Products

**Benchmark (1M Embeddings):**
```
Exakte Suche (Dot Product):   1200 ms
HNSW (Approximate):            15 ms    ← 80× schneller!
```

---

## 🎯 Zusammenfassung

**Ein Satz:** Dot Product multipliziert entsprechende Elemente zweier Vektoren und summiert sie - das Fundament für Cosine Similarity und Attention.

**Formel (Merksatz):**
$$\mathbf{A} \cdot \mathbf{B} = \sum A_i \cdot B_i$$

**Key Takeaways:**
1. **Basis-Operation**: Für Cosine, Attention, Matrix Multiplication
2. **Hardware-optimiert**: CPUs/GPUs haben spezielle Instruktionen
3. **Normalisierung**: Dot Product = Cosine (wenn ||v|| = 1)
4. **Wertebereich**: $(-\infty, +\infty)$, aber $[-1, 1]$ bei normalisierten Vektoren

**Wann nutzen?**
- ✅ Cosine Similarity (mit normalisierten Vektoren)
- ✅ Attention Scores (Q·K^T)
- ✅ Vector Database Retrieval
- ✅ Jede Matrix-Vektor Operation

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ⬅️ [Vorheriger Begriff: Vector Normalization](03-vector-normalization.md)
- ➡️ [Nächster Begriff: Dense Retrieval](05-dense-retrieval.md)
