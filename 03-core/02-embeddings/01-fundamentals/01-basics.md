# Embedding Fundamentals: Mathematische Grundlagen

## 🎯 Ziel
Verstehe die mathematischen Grundlagen von Embeddings - von Vektorräumen bis zu Distanzmetriken. Das Fundament für alle AI-Systeme, die mit Text arbeiten.

## 📖 Geschichte & Kontext
Embeddings sind die Grundlage moderner NLP-Systeme und RAG-Architektur. Von der Notwendigkeit geboren, semantische Ähnlichkeit mathematisch messbar zu machen, entwickelten sie sich zum Herzstück der Information Retrieval.

**Historische Entwicklung:**
- 1950er: Erste Versuche der automatischen Textverarbeitung
- 1990er: Latent Semantic Analysis (LSA) - Vorläufer moderner Embeddings
- 2013: Word2Vec revolutioniert NLP mit praktischen Word Embeddings
- 2018: BERT führt kontextualisierte Embeddings ein
- 2019-heute: Optimierung für Retrieval und RAG-Systeme

## 🧮 Konzept & Theorie

## Was ist ein Vektorraum?

### Intuition: Von Wörtern zu Punkten im Raum

**Problem:** Computer können nicht mit Text rechnen
```python
"Kühlschrank" + "Medikament" = ???  # Geht nicht!
```

**Lösung:** Wandle Text in Zahlen um (Vektor)
```python
"Kühlschrank" → [0.5, 0.3, 0.8]
"Medikament"  → [0.2, 0.9, 0.1]

# Jetzt kann man rechnen:
distance = sqrt((0.5-0.2)² + (0.3-0.9)² + (0.8-0.1)²) = 0.94
```

### 2D-Beispiel (vereinfacht)

Stell dir vor, wir haben nur **2 Dimensionen**:
- Dimension 1: "Wie medizinisch ist das?"
- Dimension 2: "Wie technisch ist das?"

```
Dimension 2 (Technisch)
    ↑
1.0 │        • "Motor"
    │
0.8 │  • "Kühlschrank"
    │
0.6 │
    │    • "Medikament"
0.4 │
    │
0.2 │                    • "Apfel"
    │
0.0 └─────────────────────────────> Dimension 1 (Medizinisch)
    0.0  0.2  0.4  0.6  0.8  1.0
```

**"Kühlschrank"** = [0.3, 0.8] → etwas medizinisch, sehr technisch
**"Medikament"** = [0.6, 0.4] → sehr medizinisch, etwas technisch
**"Motor"** = [0.1, 0.9] → kaum medizinisch, sehr technisch
**"Apfel"** = [0.8, 0.2] → medizinisch (gesund), kaum technisch

**Nähe im Raum = Ähnlichkeit in Bedeutung!**

---

## Dimensionen

### Was ist eine Dimension?

**Einfach gesagt:** Eine "Eigenschaft" oder "Merkmal"

**In 2D (wie oben):**
- 2 Zahlen pro Wort
- Leicht vorstellbar
- **Zu wenig für komplexe Bedeutungen!**

**In echten Embeddings:**
- **384 Dimensionen** (kleinere Models wie MiniLM)
- **768 Dimensionen** (BERT, RoBERTa)
- **1024 Dimensionen** (größere Models wie E5-large)

### Warum so viele Dimensionen?

**Analogie:** Beschreibe eine Person

**Mit 2 Eigenschaften:**
- Größe: 180cm
- Gewicht: 75kg
- **Zu wenig!** Viele verschiedene Menschen haben diese Werte.

**Mit 100 Eigenschaften:**
- Größe, Gewicht, Augenfarbe, Haarfarbe, Stimmlage, Gangart, Gestik, ...
- **Viel eindeutiger!** Kaum zwei Menschen identisch.

**Bei Text genauso:**
- 2 Dimensionen: Grobe Kategorien ("medizinisch", "technisch")
- 768 Dimensionen: Nuancierte Bedeutung, Kontext, Synonyme, Beziehungen, ...

### Was bedeuten die Dimensionen?

**Kurze Antwort:** Wir wissen es nicht genau! 🤷

**Lange Antwort:** Das Model lernt sie selbst beim Training
- Dimension 42 könnte "Grad der Formalität" sein
- Dimension 137 könnte "Zeitliche Referenz" kodieren
- Dimension 512 könnte "Sentiment" repräsentieren

**Aber:** Wir können nicht sagen "Dimension X = Eigenschaft Y"

**Wichtig:** Dimensionen sind **latent** (verborgen) und **distributed** (Bedeutung verteilt über viele Dimensionen)

### Beispiel: Realer Embedding-Vektor

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embedding = model.encode("Laborkühlschrank")

print(embedding.shape)  # (384,)
print(embedding[:10])   # Erste 10 Dimensionen:
# [ 0.234, -0.456,  0.123,  0.789, -0.234,
#   0.567, -0.890,  0.345, -0.123,  0.678]
```

**384 Zahlen** - jede zwischen ca. -1 und 1

---

## Distanzmetriken

### Wie misst man Ähnlichkeit zwischen Vektoren?

Drei Haupt-Metriken:

### 1. Cosine Similarity (am häufigsten!)

**Idee:** Misst den **Winkel** zwischen zwei Vektoren

**Formel:**
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

A · B = Skalarprodukt (Dot Product)
||A|| = Länge/Norm von A
```

**Visualisierung (2D):**
```
    B •
      ╱ ↖ kleiner Winkel
     ╱    → hohe Ähnlichkeit
    ╱
   ╱
  • A


    C •
      │
      │ 90° Winkel
      │   → keine Ähnlichkeit
      │
      • A
```

**Eigenschaften:**
- Wertebereich: -1 (entgegengesetzt) bis +1 (identisch)
- 0 = orthogonal (keine Ähnlichkeit)
- **Unabhängig von Vektorlänge!** Nur Richtung zählt

**Code:**
```python
import numpy as np

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Beispiel
vec_a = np.array([1, 2, 3])
vec_b = np.array([2, 4, 6])  # Doppelt so lang, gleiche Richtung

print(cosine_similarity(vec_a, vec_b))  # 1.0 (identische Richtung!)
```

**Warum Cosine für Embeddings?**
- ✅ Skalierungsunabhängig (Text-Länge egal)
- ✅ Gute Interpretierbarkeit
- ✅ Effizient zu berechnen
- ✅ Standard in allen Embedding-Libraries

---

### 2. Euclidean Distance (L2-Distanz)

**Idee:** Direkte **Luftlinien-Entfernung** zwischen zwei Punkten

**Formel:**
```
euclidean_distance(A, B) = √(Σ(Aᵢ - Bᵢ)²)
```

**Visualisierung (2D):**
```
       B •
        ╱│
       ╱ │ Luftlinie
      ╱  │ (Euclidean)
     ╱   │
    • A  │
```

**Code:**
```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Oder einfach:
from scipy.spatial.distance import euclidean

vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])

print(euclidean(vec_a, vec_b))  # 5.196
```

**Eigenschaften:**
- Wertebereich: 0 (identisch) bis ∞ (sehr unterschiedlich)
- **Abhängig von Vektorlänge!** Längere Texte → größere Distanzen
- Braucht **Normalisierung** für faire Vergleiche

**Wann nutzen?**
- Wenn Vektorlänge/Magnitude wichtig ist
- Bei normalisierten Embeddings (dann äquivalent zu Cosine!)
- K-Means Clustering

---

### 3. Dot Product (Skalarprodukt)

**Idee:** Summe der elementweisen Multiplikationen

**Formel:**
```
dot_product(A, B) = Σ(Aᵢ × Bᵢ)
```

**Code:**
```python
def dot_product(a, b):
    return np.dot(a, b)

vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])

print(dot_product(vec_a, vec_b))  # 1×4 + 2×5 + 3×6 = 32
```

**Eigenschaften:**
- Wertebereich: -∞ bis +∞
- **Schneller** als Cosine (keine Division/Norm-Berechnung)
- Bei **normalisierten** Vektoren = Cosine Similarity!

**Relation zu Cosine:**
```python
# Wenn Vektoren normalisiert (Länge = 1):
cosine_sim(A, B) = dot_product(A, B) / (||A|| × ||B||)
                 = dot_product(A, B) / (1 × 1)
                 = dot_product(A, B)
```

**Wann nutzen?**
- Bei vorher normalisierten Embeddings (L2-Norm)
- Für maximale Performance (z.B. Millionen Vektoren)
- In Vector Databases (FAISS, Qdrant)

---

### Vergleich der Metriken

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Embeddings generieren
emb_a = model.encode("Laborkühlschrank")
emb_b = model.encode("Medikamentenkühlschrank")
emb_c = model.encode("Pommes Frites")

# Cosine Similarity
print("Cosine A-B:", util.cos_sim(emb_a, emb_b).item())  # ~0.85 (sehr ähnlich)
print("Cosine A-C:", util.cos_sim(emb_a, emb_c).item())  # ~0.15 (unterschiedlich)

# Euclidean Distance
from scipy.spatial.distance import euclidean
print("Euclidean A-B:", euclidean(emb_a, emb_b))  # Klein (ähnlich)
print("Euclidean A-C:", euclidean(emb_a, emb_c))  # Groß (unterschiedlich)

# Dot Product
print("Dot A-B:", np.dot(emb_a, emb_b))
print("Dot A-C:", np.dot(emb_a, emb_c))
```

**Empfehlung:** **Cosine Similarity** für Text-Embeddings (Standard)

---

## Ähnlichkeit vs. Relevanz

### Wichtiger Unterschied!

**Ähnlichkeit (Similarity):** Mathematische Nähe im Vektorraum
```python
cosine_sim("Katze", "Hund") = 0.8  # Beide Tiere, ähnlich
```

**Relevanz (Relevance):** Nützlichkeit für eine Query
```python
Query: "Wie pflege ich meine Katze?"
Doc1: "Katzen sind Haustiere" → Ähnlich, aber NICHT relevant
Doc2: "Katzenpflege: Bürsten, Füttern, ..." → Sehr relevant!
```

### Das Problem

**Embedding-Models messen nur Ähnlichkeit, nicht Relevanz!**

```python
# Query
query = "Python Tutorial für Anfänger"
query_emb = model.encode(query)

# Dokumente
doc1 = "Python ist eine Programmiersprache"  # Ähnlich, wenig relevant
doc2 = "Schritt-für-Schritt Python lernen für Anfänger mit Code-Beispielen"  # Sehr relevant!

# Problem: doc1 könnte höher ranken (enthält "Python Tutorial")
# aber doc2 ist nützlicher!
```

### Lösung: Re-Ranking

1. **Retrieval:** Embedding-basierte Suche (schnell, ~100 Kandidaten)
2. **Re-Ranking:** Cross-Encoder oder LLM bewertet Relevanz (langsam, Top-10)

```python
# Stage 1: Schnelles Retrieval
candidates = vector_db.search(query_emb, top_k=100)

# Stage 2: Re-Ranking nach Relevanz
from transformers import AutoModelForSequenceClassification

cross_encoder = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')

for doc in candidates:
    relevance_score = cross_encoder.predict([query, doc.text])

top_10 = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)[:10]
```

**Siehe auch:** [05-CROSS-ENCODERS.md](05-CROSS-ENCODERS.md)

---

## Normalisierung

### Warum Normalisierung?

**Problem:** Vektoren haben unterschiedliche Längen
```python
vec_a = [1, 2, 3]     # Länge = √(1² + 2² + 3²) = 3.74
vec_b = [10, 20, 30]  # Länge = √(10² + 20² + 30²) = 37.4
# b ist 10x länger, aber gleiche Richtung!
```

**Lösung:** Normalisiere auf Einheitslänge (Länge = 1)

### L2-Normalisierung (am häufigsten)

**Formel:**
```
normalized_vector = vector / ||vector||

||vector|| = √(v₁² + v₂² + ... + vₙ²)  # L2-Norm
```

**Code:**
```python
import numpy as np

def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm

# Beispiel
vec = np.array([3, 4])
print("Original:", vec)
print("Länge:", np.linalg.norm(vec))  # 5.0

normalized = l2_normalize(vec)
print("Normalisiert:", normalized)     # [0.6, 0.8]
print("Neue Länge:", np.linalg.norm(normalized))  # 1.0
```

**Visualisierung:**
```
Vorher:
    │
  4 │     • B (3, 4)
    │    ╱
  3 │   ╱
    │  ╱
  2 │ ╱
    │╱
  0 └─────────
    0 1 2 3 4

Nachher:
    │
1.0 │   • B' (0.6, 0.8)
    │  ╱
0.8 │ ╱ <- Gleiche Richtung, Länge=1
    │╱
  0 └─────────
    0   0.6  1.0
```

### Vorteile der Normalisierung

**1. Cosine = Dot Product (schneller!)**
```python
# Ohne Normalisierung
cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Mit Normalisierung (||a|| = ||b|| = 1)
cosine = np.dot(a_normalized, b_normalized)  # Schneller!
```

**2. Fair Comparison**
- Lange vs. kurze Texte gleichberechtigt
- Nur Bedeutung zählt, nicht Länge

**3. Numerische Stabilität**
- Vermeidet Overflow/Underflow
- Konsistente Wertebereiche

### Wann normalisieren?

**Standard-Praxis:**
- ✅ Nach Embedding-Generierung (vor Speicherung)
- ✅ Query-Embeddings vor Suche
- ✅ In Vector Databases (oft automatisch)

**Nicht normalisieren wenn:**
- Magnitude wichtig ist (selten bei Text)
- Model gibt schon normalisierte Vektoren (manche tun das)

**Code-Beispiel mit Sentence-Transformers:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embeddings generieren
embeddings = model.encode(["Text 1", "Text 2"])

# Normalisieren
import numpy as np
embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Oder mit sklearn
from sklearn.preprocessing import normalize
embeddings_normalized = normalize(embeddings, norm='l2')

# Prüfen
print(np.linalg.norm(embeddings_normalized[0]))  # 1.0 ✓
```

---

## Visualisierung

### Problem: 768 Dimensionen sind nicht visualisierbar!

Wie sieht man sich Embeddings an?

### Lösung: Dimensionality Reduction

Reduziere 768 Dimensionen → 2 oder 3 Dimensionen (für Plot)

### t-SNE (t-distributed Stochastic Neighbor Embedding)

**Idee:** Behalte lokale Nachbarschaften bei

**Code:**
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Embeddings generieren
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
texts = [
    "Laborkühlschrank",
    "Medikamentenkühlschrank",
    "Blutkühlschrank",
    "Gefrierschrank",
    "Apfel",
    "Banane",
    "Orange"
]
embeddings = model.encode(texts)  # (7, 384)

# Reduziere auf 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)  # (7, 2)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

for i, txt in enumerate(texts):
    plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title('t-SNE Visualisierung von Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
```

**Erwartetes Ergebnis:**
```
Kühlschränke clustern zusammen:
• Laborkühlschrank
• Medikamentenkühlschrank
• Blutkühlschrank

Obst clustert zusammen:
• Apfel
• Banane
• Orange

Gefrierschrank irgendwo dazwischen
```

**Eigenschaften:**
- ✅ Sehr gute Cluster-Erkennung
- ✅ Schöne Visualisierungen
- ⚠️ Langsam bei vielen Punkten (>10k)
- ⚠️ Nicht-deterministisch (jeder Run anders)

---

### UMAP (Uniform Manifold Approximation and Projection)

**Alternative zu t-SNE:**
- Schneller (auch bei 100k+ Punkten)
- Deterministisch (reproduzierbar)
- Besser für große Datenmengen

**Code:**
```python
import umap

# Statt t-SNE:
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Rest wie bei t-SNE
```

---

### PCA (Principal Component Analysis)

**Einfachste Methode:**
- Linear (t-SNE/UMAP sind non-linear)
- Sehr schnell
- Gut für erste Exploration

**Code:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Wie viel Varianz wird erklärt?
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
# z.B. "Explained variance: 34%"
# → 2D kann nur 34% der Original-Info behalten
```

---

### Vergleich der Methoden

| Methode | Geschwindigkeit | Cluster-Qualität | Deterministisch | Use-Case |
|---------|----------------|------------------|-----------------|----------|
| **PCA** | Sehr schnell | Niedrig | Ja | Erste Exploration |
| **t-SNE** | Langsam | Sehr gut | Nein | Schöne Plots (<10k Punkte) |
| **UMAP** | Schnell | Sehr gut | Ja | Production, große Datenmengen |

**Empfehlung:** UMAP für meiste Anwendungen

---

## Zusammenfassung

### Key Takeaways

1. **Vektorraum:** Text wird zu Punkten in hochdimensionalem Raum
2. **Dimensionen:** 384-1024, latent und distributed (nicht interpretierbar)
3. **Cosine Similarity:** Standard-Metrik für Text-Ähnlichkeit
4. **Normalisierung:** L2-Norm auf Länge 1 (macht Cosine = Dot Product)
5. **Ähnlichkeit ≠ Relevanz:** Embeddings messen Nähe, nicht Nützlichkeit
6. **Visualisierung:** UMAP/t-SNE für 2D-Reduktion

### Mathematik-Cheat-Sheet

```python
# Cosine Similarity
cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Euclidean Distance
euclidean = np.sqrt(np.sum((a - b) ** 2))

# Dot Product
dot = np.dot(a, b)

# L2 Normalization
normalized = vector / np.linalg.norm(vector)

# Bei normalisierten Vektoren:
cosine_sim(a, b) == dot_product(a, b)  # True!
```

## 📊 Vergleiche & Varianten

### Embedding-Dimensionen im Vergleich
- **Word2Vec**: 300 Dimensionen (klassisch)
- **BERT-base**: 768 Dimensionen (Standard)
- **BERT-large**: 1024 Dimensionen (bessere Qualität)
- **Sentence-BERT**: 384-768 Dimensionen (Sentence-optimiert)

### Distanzmetriken - Wann was nutzen?
- **Cosine**: Standard für Text (Richtung wichtiger als Länge)
- **Euclidean**: Gut wenn Magnitude wichtig ist
- **Dot Product**: Schnell bei normalisierten Vektoren

## 🚀 Was du danach kannst

**Grundlagen:**
- Du verstehst die Mathematik hinter Embeddings (Vektorräume, Dimensionen)
- Du kannst Ähnlichkeitsmetriken richtig auswählen und anwenden
- Du interpretierst Embedding-Visualisierungen und erkennst Cluster

**Praktische Skills:**
- Du implementierst Distance-Funktionen von Grund auf
- Du normalisierst Embeddings für optimale Performance
- Du visualisierst hochdimensionale Embeddings in 2D/3D

**Problemverständnis:**
- Du erkennst den Unterschied zwischen Ähnlichkeit und Relevanz
- Du verstehst warum Normalisierung wichtig ist
- Du weißt wann welche Distanzmetrik zu verwenden ist

## 🔗 Weiterführende Themen

### Original Papers
- [Word2Vec](https://arxiv.org/abs/1301.3781) - Mikolov et al. 2013 (Word Embeddings)
- [BERT](https://arxiv.org/abs/1810.04805) - Devlin et al. 2018 (Contextualized Embeddings)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Reimers & Gurevych 2019

### Verwandte Kapitel
- [02-DENSE-EMBEDDINGS.md](02-DENSE-EMBEDDINGS.md) - Wie BERT & Co. funktionieren
- [03-SPARSE-EMBEDDINGS.md](03-SPARSE-EMBEDDINGS.md) - BM25 & SPLADE Alternativen
- [../core/evaluation/02-similarity-measures.md](../core/evaluation/02-similarity-measures.md) - Evaluation von Ähnlichkeit

### Nächste Schritte im Lernpfad
1. **Für RAG-Entwicklung**: [02-DENSE-EMBEDDINGS.md](02-DENSE-EMBEDDINGS.md)
2. **Für Evaluation**: [../core/evaluation/05-embedding-evaluation.md](../core/evaluation/05-embedding-evaluation.md)
3. **Für Production**: [../core/infrastructure/03-vector-databases.md](../core/infrastructure/03-vector-databases.md)

## 📚 Ressourcen

### Wissenschaftliche Papers
- [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781) - Word2Vec Grundlagen
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

### Blog Posts & Tutorials
- [StatQuest: Embeddings erklärt](https://www.youtube.com/watch?v=viZrOnJclY0)
- [3Blue1Brown: Vektoren visualisiert](https://www.youtube.com/watch?v=fNk_zzaMoSs)

### Videos & Talks
- [Word Embeddings explained](https://www.youtube.com/watch?v=ERibwqs9p38) - CodeEmporium

### Interaktive Demos
- [TensorFlow Embedding Projector](https://projector.tensorflow.org/) - Visualisiere bekannte Embeddings
- [Word2Vec Interactive](https://ronxin.github.io/wevi/) - Explore Word2Vec
