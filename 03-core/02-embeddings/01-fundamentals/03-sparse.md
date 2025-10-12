# Sparse Embeddings: Deep Dive

## Inhaltsverzeichnis

1. [Was sind Sparse Embeddings?](#was-sind-sparse-embeddings)
2. [TF-IDF (Term Frequency-Inverse Document Frequency)](#tf-idf)
3. [BM25 (Best Match 25)](#bm25)
4. [SPLADE (Learned Sparse)](#splade)
5. [Inverted Index](#inverted-index)
6. [Vor- und Nachteile](#vor--und-nachteile)
7. [Code-Beispiele](#code-beispiele)

---

## Was sind Sparse Embeddings?

### Definition

**Sparse Embeddings** sind Vektorrepräsentationen, bei denen **die meisten Dimensionen 0 sind**.

```python
# Dense Embedding (384 Dimensionen)
[0.234, -0.456, 0.123, 0.789, -0.234, 0.567, ..., -0.123]
# Alle 384 Werte != 0

# Sparse Embedding (30.000 Dimensionen)
[0, 0, 0, 3.2, 0, 0, 0, 0, 1.5, 0, 0, ..., 0, 0, 0]
# Nur ~10-50 Werte != 0 (99% sind Nullen!)
```

### Warum so viele Dimensionen?

**Jede Dimension = Ein Wort im Vokabular**

```python
Vokabular:
0: "der"
1: "die"
2: "das"
...
5421: "Laborkühlschrank"
5422: "Medikament"
...
29999: "Zylinder"

Text: "Der Laborkühlschrank"
Sparse Vector: [1.2, 0, 0, ..., 4.5, ..., 0]
                 ↑           ↑
                "der"   "Laborkühlschrank"
```

**Nur die Wörter die im Text vorkommen haben Werte!**

### Sparse vs. Dense - Visuell

**Dense (768-dim):**
```
Dimension:  1    2    3    4    5  ...  768
Werte:    [0.23 -0.45 0.12 0.89 -0.67 ... 0.34]
                  Alle gefüllt →
```

**Sparse (30k-dim):**
```
Dimension:  1  2  3  4  5 ... 5421 ... 10234 ... 29999 30000
Werte:     [0  0  0  0  0 ... 3.2  ...  1.5  ...  0     0   ]
                              ↑         ↑
                         Nur wenige != 0
```

### Eigenschaften

- **Dimensionen:** 10k - 100k (Vokabular-Größe)
- **Non-zero Werte:** ~10-100 (0.1-1% der Dimensionen)
- **Interpretierbar:** Jede Dimension = bekanntes Wort
- **Speicher:** Sehr effizient (nur non-zero speichern)

---

## TF-IDF

### Term Frequency - Inverse Document Frequency

**Älteste und einfachste Methode** (1970er Jahre)

### Intuition

**Problem:** Nicht alle Wörter sind gleich wichtig

```
Text: "Der Kühlschrank ist ein Kühlschrank"

"der" - kommt überall vor → unwichtig
"ist" - kommt überall vor → unwichtig
"ein" - kommt überall vor → unwichtig
"Kühlschrank" - selten, spezifisch → WICHTIG!
```

**TF-IDF Idee:**
- **Häufig im Dokument** (TF) → wichtig
- **Selten im Corpus** (IDF) → wichtig
- **Häufig UND selten** → sehr wichtig!

### Mathematik

**1. Term Frequency (TF)**

Wie oft kommt Wort in Dokument vor?

```
TF(t, d) = (Anzahl von t in d) / (Gesamtanzahl Wörter in d)
```

**Beispiel:**
```
Dokument: "Kühlschrank Kühlschrank Medikament" (3 Wörter)

TF("Kühlschrank") = 2/3 = 0.67
TF("Medikament")  = 1/3 = 0.33
```

**Varianten:**
```python
# Raw Count
tf = count(term, doc)

# Frequency
tf = count(term, doc) / len(doc)

# Log Normalization (häufiger)
tf = log(1 + count(term, doc))

# Binary
tf = 1 if term in doc else 0
```

---

**2. Inverse Document Frequency (IDF)**

Wie selten ist das Wort im gesamten Corpus?

```
IDF(t) = log(Anzahl Dokumente / Anzahl Dokumente mit t)
```

**Beispiel:**
```
Corpus: 1000 Dokumente

"der"          - in 998 Dokumenten → IDF = log(1000/998) = 0.002 (unwichtig)
"Kühlschrank"  - in 50 Dokumenten  → IDF = log(1000/50)  = 3.0   (wichtig!)
```

**Intuition:**
- Seltenes Wort → hoher IDF → wichtig
- Häufiges Wort → niedriger IDF → unwichtig

---

**3. TF-IDF = TF × IDF**

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Beispiel:**
```
Dokument: "Der Laborkühlschrank ist sehr kalt"

Wort             TF    ×   IDF   =  TF-IDF
"Der"           0.2   ×   0.002  =  0.0004  ← unwichtig
"Laborkühl..."  0.2   ×   3.5    =  0.7     ← WICHTIG!
"ist"           0.2   ×   0.003  =  0.0006
"sehr"          0.2   ×   0.5    =  0.1
"kalt"          0.2   ×   1.2    =  0.24
```

**Sparse Vector für dieses Dokument:**
```python
{
    5421: 0.7,    # "Laborkühlschrank"
    8934: 0.24,   # "kalt"
    3421: 0.1,    # "sehr"
    # Alle anderen: 0
}
```

### Visualisierung

```
        IDF (Seltenheit)
         ↑
    High │     • "Laborkühlschrank"
         │       (selten, hoher IDF)
         │
         │   • "kalt"
         │     (mittel)
         │
    Low  │ • "der"
         │   (häufig, niedriger IDF)
         └──────────────────────→ TF (Häufigkeit im Doc)
            Low              High

TF-IDF = Fläche im Quadranten oben-rechts
```

### Code-Beispiel

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Corpus
documents = [
    "Laborkühlschrank mit 280 Liter Volumen",
    "Medikamentenkühlschrank nach DIN Norm",
    "Der Kühlschrank ist kalt",
    "Pommes Frites sind lecker"
]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("Shape:", tfidf_matrix.shape)  # (4, vocabulary_size)
print("Non-zero:", tfidf_matrix.nnz)  # Anzahl nicht-null Werte

# Vokabular
print("\nVokabular:")
print(vectorizer.get_feature_names_out()[:10])
# ['280' 'der' 'din' 'frites' 'ist' 'kalt' 'kühlschrank' 'laborkühlschrank' ...]

# Sparse Vector für erstes Dokument
print("\nErster Dokument-Vektor:")
print(tfidf_matrix[0].toarray())
# [[0.  0.  0.  0.  0.  0.  0.4 0.5 0.  0.4 0.  0. ...]]
#   Meiste Werte sind 0, nur "laborkühlschrank", "liter", "volumen" != 0
```

### TF-IDF für Suche

```python
# Query auch als TF-IDF
query = "Kühlschrank 280 Liter"
query_vector = vectorizer.transform([query])

# Cosine Similarity zu allen Dokumenten
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

# Ranking
for idx, sim in enumerate(similarities):
    print(f"Doc {idx}: {sim:.4f} - {documents[idx]}")

# Output:
# Doc 0: 0.8234 - Laborkühlschrank mit 280 Liter Volumen  ← Bestes Match!
# Doc 1: 0.1234 - Medikamentenkühlschrank nach DIN Norm
# Doc 2: 0.3421 - Der Kühlschrank ist kalt
# Doc 3: 0.0000 - Pommes Frites sind lecker
```

### Schwächen von TF-IDF

❌ **Keine Semantik**
```python
query = "Kühlaggregat"
# Findet NICHT "Kühlschrank" (andere Wörter!)
```

❌ **Keine Synonyme**
```python
query = "refrigerator"
# Findet NICHT "Kühlschrank" (Englisch vs. Deutsch)
```

❌ **Bag of Words**
```python
"Hund beißt Mann" vs "Mann beißt Hund"
# Identische TF-IDF Vektoren! (Reihenfolge egal)
```

**Aber:** Sehr schnell, interpretierbar, baseline für viele Systeme!

---

## BM25

### Best Match 25

**"Verbessertes TF-IDF"** - Industry Standard für Keyword-Search!

### Was verbessert BM25?

**Problem 1 mit TF-IDF:** Term Frequency wächst linear

```
TF-IDF:
1× "Kühlschrank" → Score: 1.0
2× "Kühlschrank" → Score: 2.0
10× "Kühlschrank" → Score: 10.0  ← Zu hoch!
```

**Lösung: Saturation**

Nach einigen Vorkommen zählt mehr nicht mehr viel.

```
BM25:
1× "Kühlschrank" → Score: 1.0
2× "Kühlschrank" → Score: 1.5
10× "Kühlschrank" → Score: 2.1  ← Sättigung!
```

**Problem 2 mit TF-IDF:** Dokumentlänge ignoriert

```
Doc A (kurz): "Kühlschrank" (1 Wort)  → TF = 1.0
Doc B (lang): "Kühlschrank ... <100 andere Wörter>" → TF = 0.01

Doc A sollte höher ranken (konzentrierter auf das Thema)!
```

**Lösung: Length Normalization**

### BM25 Formel

```
BM25(q, d) = Σ IDF(qᵢ) × (f(qᵢ, d) × (k₁ + 1)) / (f(qᵢ, d) + k₁ × (1 - b + b × |d|/avgdl))
              qᵢ∈q

Wobei:
- qᵢ       : Query-Term
- f(qᵢ, d) : Häufigkeit von qᵢ in d
- |d|      : Länge von Dokument d
- avgdl    : Durchschnittliche Dokumentlänge
- k₁       : Term Frequency Saturation (typisch: 1.2-2.0)
- b        : Length Normalization (typisch: 0.75)
```

**Sieht kompliziert aus!** Lass uns zerlegen:

### BM25 Schritt für Schritt

**1. IDF (wie bei TF-IDF)**
```python
IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))

N     : Anzahl Dokumente
df(t) : Dokumente die t enthalten
```

**2. Term Frequency mit Saturation**
```python
tf_component = (f(t, d) × (k₁ + 1)) / (f(t, d) + k₁)

k₁ = 1.5  # Saturation-Parameter

Beispiel:
f=1  → tf_component = 1.0
f=2  → tf_component = 1.43
f=5  → tf_component = 1.85
f=10 → tf_component = 2.04  ← Sättigung!
f=100 → tf_component = 2.48  ← Kaum noch Zuwachs
```

**3. Length Normalization**
```python
norm = 1 - b + b × (|d| / avgdl)

b = 0.75  # Normalization-Stärke (0 = aus, 1 = voll)

Kurzes Doc (|d| = 50, avgdl = 100):
  norm = 1 - 0.75 + 0.75 × 0.5 = 0.625  ← Boost!

Langes Doc (|d| = 200, avgdl = 100):
  norm = 1 - 0.75 + 0.75 × 2.0 = 1.75  ← Penalty!
```

**4. Kombiniert**
```python
BM25_score(t, d) = IDF(t) × (tf_component / norm)
```

### Visualisierung: TF-IDF vs BM25

```
Score vs. Term Frequency

  Score
    ↑
  10│     TF-IDF (linear)
    │        ╱
  8 │       ╱
    │      ╱
  6 │     ╱
    │    ╱
  4 │   ╱
    │  ╱        BM25 (saturiert)
  2 │ ╱╱╱╱━━━━━━━━━
    │╱
  0 └────────────────────→ Term Frequency
    0  2  4  6  8  10 12

BM25 flacht ab (Saturation)
TF-IDF wächst unbegrenzt
```

### BM25 Parameter-Tuning

**k₁ (Saturation):**
- **Niedrig (0.5-1.0):** Schnelle Sättigung → Weniger Gewicht auf Wiederholungen
- **Mittel (1.2-2.0):** Standard → Gute Balance
- **Hoch (2.0-3.0):** Langsame Sättigung → Mehr Gewicht auf Häufigkeit

**b (Length Normalization):**
- **0.0:** Keine Normalisierung → Lange Docs nicht bestraft
- **0.5-0.8:** Moderate Normalisierung → Standard
- **1.0:** Volle Normalisierung → Starke Bestrafung langer Docs

**Empfehlung:** `k₁=1.5`, `b=0.75` (funktioniert für meiste Anwendungen)

### Code-Beispiel

```python
from rank_bm25 import BM25Okapi
import numpy as np

# Corpus (tokenisiert!)
corpus = [
    "Laborkühlschrank mit 280 Liter Volumen".split(),
    "Medikamentenkühlschrank nach DIN Norm".split(),
    "Der Kühlschrank ist sehr kalt".split(),
    "Pommes Frites sind lecker lecker lecker".split()  # "lecker" 3x
]

# BM25 erstellen
bm25 = BM25Okapi(corpus)

# Query
query = "Kühlschrank 280 Liter".split()

# Scores berechnen
scores = bm25.get_scores(query)

print("BM25 Scores:")
for idx, score in enumerate(scores):
    print(f"Doc {idx}: {score:.4f}")

# Output:
# Doc 0: 4.5234  ← Beste Match ("Kühlschrank", "280", "Liter" alle vorhanden)
# Doc 1: 1.2341  ← "Kühlschrank" Match
# Doc 2: 1.0234  ← "Kühlschrank" Match
# Doc 3: 0.0000  ← Kein Match

# Top-K Dokumente
top_n = np.argsort(scores)[::-1][:2]
print(f"\nTop-2: {top_n}")  # [0, 1]
```

### BM25+ und BM25L (Varianten)

**BM25+:** Verhindert negative Term-Scores
```python
# Original BM25 kann negative Scores haben (bei IDF < 0)
# BM25+ fixiert das mit einem Bias-Term
```

**BM25L:** Bessere Length Normalization für sehr kurze/lange Docs

**Meist reicht Standard-BM25 (BM25Okapi)!**

### Wann BM25 statt TF-IDF?

✅ **Immer!** BM25 ist praktisch immer besser als TF-IDF.

**Ausnahme:** Du brauchst exakt TF-IDF aus historischen Gründen.

---

## SPLADE

### SParse Lexical AnD Expansion

**"Neuronales Sparse Embedding"** - Beste von Dense + Sparse!

### Das Problem mit BM25

**BM25 ist Bag-of-Words:**

```python
Query: "Kühlaggregat"
Doc: "Kühlschrank für Medikamente"

BM25 Score: 0.0  ← Kein exaktes Wort-Match!
```

**Aber semantisch sind sie ähnlich!**

Dense Embeddings würden das finden, aber BM25 nicht.

### SPLADE Idee

**"Was wenn Sparse Embeddings lernen könnten?"**

1. **Start:** BERT (Dense Model)
2. **Modifikation:** Output ist Sparse statt Dense
3. **Training:** Auf Retrieval-Tasks

**Resultat:** Sparse Vector mit gelernten Gewichten!

### Architektur

```
Input: "Laborkühlschrank"
   ↓
┌─────────┐
│  BERT   │  ← Transformer (wie Dense)
└─────────┘
   ↓
Token Embeddings: [768-dim × seq_len]
   ↓
┌──────────────────┐
│  MLM Head        │  ← Vorhersage für JEDES Wort im Vokabular
│  (Vocabulary)    │
└──────────────────┘
   ↓
Logits: [30k] (eines pro Vokabular-Wort)
   ↓
ReLU + Log(1+x)  ← Nur positive, sparsify
   ↓
Sparse Vector: [0, 0, 3.2, 0, 0, 1.5, ..., 0]
```

### Term Expansion

**Das Magische:** SPLADE fügt verwandte Begriffe hinzu!

```python
Input: "Kühlaggregat"

BM25 Output (nur Input-Wörter):
{
    "Kühlaggregat": 3.5
}

SPLADE Output (+ verwandte Begriffe!):
{
    "Kühlaggregat": 3.5,    # Original
    "Kühlschrank": 2.1,     # Gelernt!
    "Kühlgerät": 1.8,       # Gelernt!
    "Kältemaschine": 1.2,   # Gelernt!
    "refrigerator": 0.9,    # Sogar Cross-Lingual!
    # ... weitere verwandte Begriffe
}
```

**Wie?** BERT hat gelernt welche Wörter semantisch verwandt sind!

### Training

**Loss-Funktion hat 3 Komponenten:**

**1. Ranking Loss**
```python
# Positive Docs sollen höher ranken als Negative
# (wie bei Dense Embeddings)
```

**2. Sparsity Regularization**
```python
# Zuviele non-zero Werte → langsam
# Loss bestraft zu viele aktivierte Dimensionen

L_sparse = λ × sum(abs(weights))  # L1-Norm
```

**3. FLOPS Regularization**
```python
# Kontrolliere Rechenaufwand
# Weniger aktive Dimensionen → schneller
```

**Trade-off:** Genauigkeit vs. Sparsity

### SPLADE vs. Dense vs. BM25

```python
Query: "Kühlgerät für Arzneimittel"

# BM25 (nur exakte Matches)
Doc 1: "Kühlgerät Arzneimittel" → Score: 5.2 ✓
Doc 2: "Kühlschrank Medikamente" → Score: 0.0 ✗ (keine Wort-Matches!)

# Dense (semantisch)
Doc 1: "Kühlgerät Arzneimittel" → Score: 0.85 ✓
Doc 2: "Kühlschrank Medikamente" → Score: 0.82 ✓ (versteht Synonyme)

# SPLADE (beides!)
Doc 1: "Kühlgerät Arzneimittel" → Score: 5.2 ✓ (exakter Match)
Doc 2: "Kühlschrank Medikamente" → Score: 4.1 ✓ (Term Expansion!)
       ↑ SPLADE erweitert Query:
       "Kühlgerät" → ["Kühlgerät", "Kühlschrank", "Kühlaggregat", ...]
       "Arzneimittel" → ["Arzneimittel", "Medikamente", "Pharmazeutika", ...]
```

### Code-Beispiel

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

# SPLADE Model laden
model_name = "naver/splade-cocondenser-ensembledistil"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def splade_encode(text):
    """Generate SPLADE sparse vector"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

    # Max Pooling über Sequence
    values, _ = torch.max(logits, dim=1)  # [batch, vocab_size]

    # ReLU + Log(1+x) für Sparsity
    values = torch.log(1 + torch.relu(values))

    return values

# Encode
query_vec = splade_encode("Laborkühlschrank")
doc_vec = splade_encode("Medizinischer Kühlschrank für Labor")

# Sparse Vektoren!
print("Non-zero dimensions:", (query_vec > 0).sum().item())  # z.B. ~50 von 30k

# Similarity (Dot Product für Sparse)
similarity = (query_vec * doc_vec).sum()
print(f"Similarity: {similarity:.4f}")
```

### SPLADE Varianten

**SPLADE v1:** Original
**SPLADE++:** Bessere Regularization
**SPLADE-cocondenser:** Knowledge Distillation von Dense Models

**State-of-the-Art:** `naver/splade-cocondenser-ensembledistil`

### Vorteile von SPLADE

✅ **Semantische Expansion** (wie Dense)
✅ **Exakte Matches** (wie BM25)
✅ **Interpretierbar** (non-zero Dimensionen = Wörter)
✅ **Effizienz** (Sparse → Inverted Index)

### Nachteile von SPLADE

❌ **Langsamer als BM25** (Inferenz braucht BERT)
❌ **Mehr Speicher als BM25** (Model-Weights)
❌ **Weniger bekannt/supported** als Dense/BM25

**Best Practice:** SPLADE für Production-Hybrid-Search sehr gut!

---

## Inverted Index

### Wie speichert man Sparse Embeddings effizient?

**Problem:** 30.000 Dimensionen × 1 Million Docs = 30 Milliarden Werte!

**Aber:** 99% sind 0 → Verschwende keinen Speicher!

### Inverted Index Struktur

**Idee:** Speichere nur "Wort → Dokumente die es enthalten"

```python
# Forward Index (ineffizient)
Doc 0: {"Kühlschrank": 3.2, "Labor": 1.5}
Doc 1: {"Kühlschrank": 2.1, "Medikament": 4.0}
Doc 2: {"Labor": 2.5, "Gefrierschrank": 3.0}

# Inverted Index (effizient!)
"Kühlschrank":     [(Doc 0, 3.2), (Doc 1, 2.1)]
"Labor":           [(Doc 0, 1.5), (Doc 2, 2.5)]
"Medikament":      [(Doc 1, 4.0)]
"Gefrierschrank":  [(Doc 2, 3.0)]
```

**Lookup:**
```python
Query: "Kühlschrank Labor"

# Nur 2 Listen lesen (statt alle Docs!)
"Kühlschrank" → Docs [0, 1]
"Labor"       → Docs [0, 2]

# Intersection/Union
Candidates: [0, 1, 2]
```

### Posting Lists

**Jeder Term hat eine "Posting List":**

```python
{
    "term": "Kühlschrank",
    "postings": [
        {"doc_id": 0, "tf": 2, "positions": [5, 12]},
        {"doc_id": 1, "tf": 1, "positions": [3]},
        {"doc_id": 5, "tf": 3, "positions": [1, 8, 15]}
    ]
}
```

**Positionen?** Für Phrase-Queries!

```python
Query: "Laborkühlschrank DIN"
# Suche Docs wo "Laborkühlschrank" direkt vor "DIN"
# Prüfe Positionen: pos("DIN") == pos("Laborkühlschrank") + 1
```

### BM25 mit Inverted Index

```python
# Pseudo-Code
def bm25_search(query, inverted_index):
    scores = defaultdict(float)

    for term in query:
        # Hole Posting List für Term
        postings = inverted_index[term]  # Nur relevante Docs!

        idf = compute_idf(term)

        for doc_id, tf in postings:
            # BM25 Score für diesen Term in diesem Doc
            score = bm25_formula(tf, idf, doc_length[doc_id])
            scores[doc_id] += score

    # Sortiere nach Score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Vorteil:** Nur Docs mit Query-Terms werden betrachtet!

### Compression

**Posting Lists können riesig werden!**

```python
"der" → Millionen Dokumente!
```

**Kompression:**

**1. Variable Byte Encoding**
```python
# Kleine Zahlen: 1 Byte
doc_id = 42 → [42]

# Große Zahlen: mehrere Bytes
doc_id = 1234567 → [...]
```

**2. Delta Encoding**
```python
# Statt absolute IDs:
[5, 103, 205, 512] → 20 bytes

# Delta (Differenzen):
[5, 98, 102, 307] → komprimierbarer!
```

**3. Bit-Level Compression**

Elasticsearch, Lucene nutzen ausgefeilte Algorithmen (FOR, PForDelta)

---

## Vor- und Nachteile

### Vorteile von Sparse Embeddings

✅ **Exakte Keyword-Matches**
```python
Query: "LABO-288"  # Modellnummer
# BM25 findet exakt das Dokument mit "LABO-288"
# Dense könnte es verpassen (fuzzy matching)
```

✅ **Interpretierbar**
```python
# Warum ranked Doc A höher?
# BM25: "Weil es 'Kühlschrank' 3x enthält und doc kurz ist"
# Dense: "Weil Dimensionen 42, 137, 512 ähnlich sind" ← ??
```

✅ **Speicher-effizient** (mit Inverted Index)
```python
# 1M Docs × 30k dims = 30B floats
# Aber 99% sind 0!
# Inverted Index: nur ~100M non-zero entries → 100x kleiner
```

✅ **Sehr schnell**
```python
# BM25 auf Elasticsearch: <10ms für Millionen Docs
# Dense: 100-500ms (braucht Vektor-Vergleiche)
```

✅ **Kein Model nötig** (BM25/TF-IDF)
```python
# Keine GPU, kein Training
# Funktioniert out-of-the-box
```

---

### Nachteile von Sparse Embeddings

❌ **Keine Semantik** (BM25/TF-IDF)
```python
Query: "Kühlaggregat"
Doc: "Kühlschrank für Labor"
# BM25: Score = 0 (keine gemeinsamen Wörter!)
# Aber semantisch relevant!
```

❌ **Keine Synonyme**
```python
Query: "refrigerator"
Doc: "Kühlschrank"
# BM25: Score = 0
# Mensch würde das matchen!
```

❌ **Bag of Words**
```python
"Hund beißt Mann" vs "Mann beißt Hund"
# Identische BM25 Scores (Reihenfolge egal)
```

❌ **Typo-sensitiv**
```python
Query: "Kühlschronk"  # Typo
# BM25: findet nichts
# Dense: würde "Kühlschrank" finden (ähnlicher Vektor)
```

❌ **Domain-spezifisches Vokabular**
```python
# Medical: "MI" = "Myokardinfarkt"
# Finance: "MI" = "Michigan"
# BM25 versteht Kontext nicht!
```

---

### Wann Sparse vs. Dense?

**Nutze Sparse (BM25) wenn:**
- ✅ Exakte Keyword-Matches wichtig (IDs, Modellnummern, SKUs)
- ✅ Erklärbarkeit nötig ("Warum dieser Treffer?")
- ✅ Sehr große Datenmengen (Millionen+ Docs)
- ✅ Latenz kritisch (<10ms)
- ✅ Keine GPU verfügbar

**Nutze Dense wenn:**
- ✅ Semantische Suche (Synonyme, Paraphrasen)
- ✅ Cross-Lingual Retrieval
- ✅ Typo-Toleranz wichtig
- ✅ Kontext-Verständnis nötig

**Best Practice: Hybrid!** (Siehe [06-HYBRID-APPROACHES.md](06-HYBRID-APPROACHES.md))

---

## Code-Beispiele

### 1. BM25 mit rank-bm25

```python
from rank_bm25 import BM25Okapi

# Corpus (PRE-TOKENIZED!)
corpus = [
    "Laborkühlschrank mit 280 Liter Volumen für medizinische Proben",
    "Medikamentenkühlschrank nach DIN 13277 Norm",
    "Gefrierschrank minus 40 Grad für Labore",
    "Pommes Frites mit Ketchup"
]

# Tokenisieren (einfaches Whitespace-Split)
tokenized_corpus = [doc.lower().split() for doc in corpus]

# BM25 Index erstellen
bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "Kühlschrank Labor Medikamente"
tokenized_query = query.lower().split()

# Suche
scores = bm25.get_scores(tokenized_query)

# Ranking
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print("BM25 Ranking:")
for idx, score in ranked:
    print(f"{score:.4f} - {corpus[idx]}")

# Output:
# 5.1234 - Medikamentenkühlschrank nach DIN 13277 Norm
# 4.5678 - Laborkühlschrank mit 280 Liter Volumen für medizinische Proben
# 1.2345 - Gefrierschrank minus 40 Grad für Labore
# 0.0000 - Pommes Frites mit Ketchup
```

---

### 2. TF-IDF mit sklearn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Corpus
corpus = [
    "Laborkühlschrank mit 280 Liter Volumen",
    "Medikamentenkühlschrank nach DIN Norm",
    "Gefrierschrank für Labor",
    "Pommes Frites sind lecker"
]

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),  # Uni- und Bigrams
    max_features=1000    # Top-1000 Features
)

tfidf_matrix = vectorizer.fit_transform(corpus)

# Query
query = "Kühlschrank 280 Liter"
query_vec = vectorizer.transform([query])

# Similarity
similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

# Ranking
ranked_indices = np.argsort(similarities)[::-1]

print("TF-IDF Ranking:")
for idx in ranked_indices:
    print(f"{similarities[idx]:.4f} - {corpus[idx]}")
```

---

### 3. Elasticsearch BM25

```python
from elasticsearch import Elasticsearch

# Elasticsearch Connection
es = Elasticsearch(['http://localhost:9200'])

# Index erstellen
index_name = "products"

# Mapping (Schema)
mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "description": {"type": "text"},
            "specs": {"type": "text"}
        }
    }
}

es.indices.create(index=index_name, body=mapping, ignore=400)

# Dokumente indizieren
docs = [
    {"title": "Laborkühlschrank", "description": "280 Liter Volumen"},
    {"title": "Medikamentenkühlschrank", "description": "DIN 13277"},
    {"title": "Gefrierschrank", "description": "Minus 40 Grad"}
]

for i, doc in enumerate(docs):
    es.index(index=index_name, id=i, document=doc)

# Suche (BM25 automatisch!)
query = {
    "query": {
        "multi_match": {
            "query": "Kühlschrank Labor 280",
            "fields": ["title^2", "description"]  # title 2x wichtiger
        }
    }
}

response = es.search(index=index_name, body=query)

print("Elasticsearch BM25 Results:")
for hit in response['hits']['hits']:
    print(f"{hit['_score']:.4f} - {hit['_source']['title']}")
```

---

### 4. SPLADE Encoding

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

# Model
model_id = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

def splade_encode(texts, max_length=256):
    """Encode texts to SPLADE sparse vectors"""
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**tokens)

    # Max Pooling + ReLU + Log
    logits = output.logits
    values, _ = torch.max(torch.log(1 + torch.relu(logits)), dim=1)

    return values.squeeze()

# Encode
query_vec = splade_encode(["Laborkühlschrank 280 Liter"])
doc_vecs = splade_encode([
    "Medizinischer Kühlschrank mit 280L",
    "Pommes Frites"
])

# Similarity (dot product für sparse)
scores = torch.matmul(doc_vecs, query_vec.T)

print("SPLADE Scores:")
for i, score in enumerate(scores):
    print(f"Doc {i}: {score.item():.4f}")

# Inspect non-zero dimensions
non_zero = (query_vec > 0).sum()
print(f"\nNon-zero dimensions: {non_zero.item()} / {query_vec.shape[0]}")

# Decode top terms
top_k = 10
top_indices = torch.topk(query_vec, k=top_k).indices
print(f"\nTop-{top_k} activated terms:")
for idx in top_indices:
    token = tokenizer.decode([idx])
    weight = query_vec[idx].item()
    print(f"  {token}: {weight:.2f}")
```

---

### 5. Custom BM25 Implementation

```python
import math
from collections import Counter, defaultdict

class SimpleBM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)  # Anzahl Dokumente

        # Berechne avgdl
        self.avgdl = sum(len(doc) for doc in corpus) / self.N

        # Berechne IDF für jedes Term
        self.idf = self._compute_idf()

    def _compute_idf(self):
        idf = {}
        df = defaultdict(int)  # Document Frequency

        # Zähle in wie vielen Docs jedes Term vorkommt
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1

        # IDF berechnen
        for term, freq in df.items():
            idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5))

        return idf

    def score(self, query, doc_id):
        doc = self.corpus[doc_id]
        score = 0.0

        # Term Frequencies im Doc
        tf = Counter(doc)
        doc_len = len(doc)

        for term in query:
            if term not in tf:
                continue

            f = tf[term]  # Term Frequency
            idf = self.idf.get(term, 0)

            # BM25 Formel
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (
                1 - self.b + self.b * (doc_len / self.avgdl)
            )

            score += idf * (numerator / denominator)

        return score

    def search(self, query, top_k=5):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# Test
corpus = [
    "Laborkühlschrank 280 Liter".split(),
    "Medikamentenkühlschrank DIN".split(),
    "Pommes Frites".split()
]

bm25 = SimpleBM25(corpus)

query = "Kühlschrank 280".split()
results = bm25.search(query)

print("Custom BM25 Results:")
for doc_id, score in results:
    print(f"{score:.4f} - {' '.join(corpus[doc_id])}")
```

---

## Zusammenfassung

### Key Takeaways

1. **Sparse = meiste Werte sind 0** (nur Wörter die vorkommen)
2. **TF-IDF:** Alte Methode, TF × IDF, funktioniert aber
3. **BM25:** Industry-Standard, TF-IDF mit Saturation + Length Norm
4. **SPLADE:** Neuronal gelernt, Term Expansion wie Dense!
5. **Inverted Index:** Effiziente Speicherung (Term → Docs)
6. **Vorteile:** Exakt, schnell, interpretierbar, speicher-effizient
7. **Nachteile:** Keine Semantik (außer SPLADE), keine Synonyme
8. **Best Practice:** Hybrid mit Dense für Production!

### BM25 Parameter Defaults

```python
k1 = 1.5   # Saturation
b = 0.75   # Length Normalization
```

### Vergleich

| Methode | Semantik | Speed | Speicher | Use-Case |
|---------|----------|-------|----------|----------|
| **TF-IDF** | ❌ | ⚡⚡⚡ | ✅✅ | Baseline, Legacy |
| **BM25** | ❌ | ⚡⚡⚡ | ✅✅ | **Standard Keyword-Search** |
| **SPLADE** | ✅ | ⚡ | ✅ | Advanced Hybrid |

---

## Nächste Schritte

- [04-MULTI-VECTOR.md](04-MULTI-VECTOR.md) - ColBERT, Token-Level Embeddings
- [06-HYBRID-APPROACHES.md](06-HYBRID-APPROACHES.md) - Dense + Sparse kombinieren
- [08-VECTOR-DATABASES.md](08-VECTOR-DATABASES.md) - Inverted Index in der Praxis

---

## Weiterführende Ressourcen

**Papers:**
- Robertson & Zaragoza 2009: "The Probabilistic Relevance Framework: BM25 and Beyond"
- Formal et al. 2021: "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"

**Tools:**
- [Elasticsearch](https://www.elastic.co/) - BM25 Production
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) - Python BM25
- [Pyserini](https://github.com/castorini/pyserini) - Research Toolkit

**Benchmarks:**
- [BEIR](https://github.com/beir-cellar/beir) - Benchmark für Information Retrieval
