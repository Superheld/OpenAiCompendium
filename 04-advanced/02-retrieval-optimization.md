# Retrieval Optimization: Von 60% zu 85% Recall

## ❓ Das Problem (Problem-First)

**Ohne Retrieval-Optimization geht folgendes schief:**
- **Schlechtes Chunking**: "Der LABO-288 Kühlschrank... [500 Wörter später] ...hat 280 Liter Volumen" → Embedding verliert Zusammenhang zwischen Modell und Kapazität
- **Kein Re-Ranking**: Dense findet "Python ist eine Programmiersprache" höher als "Python Tutorial für Anfänger" → Ähnlichkeit ≠ Relevanz
- **Nur Dense ODER Sparse**: Query "LABO-288 Kühlschrank 280L" → Dense versteht Semantik, verliert aber Modellnummer; BM25 findet Modellnummer, versteht aber nicht "refrigerator" als Synonym

**Die zentrale Frage:**
Wie baue ich eine Retrieval-Pipeline die sowohl exakte Matches findet (BM25) als auch semantisch versteht (Dense) UND dann die wirklich relevanten Docs identifiziert (Re-Ranking)?

**Beispiel-Szenario:**
```python
Query: "Laborkühlschrank 280 Liter mit Temperaturalarm"

# Naive Approach (nur Dense):
results = [
    "Laborkühlschrank mit Volumen",      # Score: 0.75 (ähnlich, aber vage)
    "Kühlschrank 280L Alarm-System",     # Score: 0.73 (relevant!)
    "Labor-Equipment Kühlung",           # Score: 0.71 (zu fuzzy)
]
# Problem: Reihenfolge suboptimal, #3 ist nicht relevant!

# Optimized Approach (Chunking + Hybrid + Re-Ranking):
results = [
    "Laborkühlschrank 280L mit Temperaturalarm", # Score: 0.94 (perfekt!)
    "Medikamentenkühlschrank 280 Liter Alarm",   # Score: 0.89 (sehr relevant)
    "LABO-288 Kühlschrank Alarmfunktion",        # Score: 0.85 (relevant)
]
# → +20% Precision!
```

## 🎯 Lernziele

Nach diesem Kapitel kannst du:
- [ ] Du verstehst WARUM Chunking kritisch ist und wählst richtige Strategie für deinen Doc-Typ
- [ ] Du implementierst Two-Stage Retrieval (schnelles Dense → präzises Cross-Encoder Re-Ranking)
- [ ] Du baust Hybrid Search (Dense + BM25) und tunest alpha-Parameter für deinen Use Case

## 🧠 Intuition zuerst (Scaffolded Progression)

### Alltagsanalogie: Buch-Suche in Bibliothek

**Beispiel: Du suchst Info über "Quantenphysik Experimente"**

**Naive Suche (nur ein Ansatz):**
```
1. Index-Suche (= Sparse):
   → Findet Bücher mit exakt "Quantenphysik" + "Experimente"
   → Verpasst: "Quantenmechanik Tests", "Teilchenphysik Versuche"

2. Bibliothekar fragt (= Dense):
   → Versteht: "Ah, moderne Physik Experimente!"
   → Findet auch Synonyme
   → Aber: Zu viele Treffer (auch "Astronomie" etc.)
```

**Optimierte Suche (Multi-Stage):**
```
1. Schnelle Vorauswahl:
   Bibliothekar überfliegt Regale → 100 Kandidaten-Bücher (2 Minuten)

2. Index-Kombination:
   Prüft Index für exakte Keywords → boosted relevante Kandidaten

3. Detaillierte Prüfung:
   Liest Inhaltsverzeichnis von Top-20 genau → findet beste 5 (10 Minuten)

→ Beste Qualität bei akzeptabler Zeit!
```

### Visualisierung: Die Retrieval-Pipeline

```
                Retrieval-Qualität
                     ↑
    Cross-Encoder    │ ⭐⭐⭐⭐⭐
    Re-Ranking       │    │
                     │    │  Hybrid
    Dense + BM25     │    │  (Dense+Sparse)
    Hybrid           │    ├──────────── ⭐⭐⭐⭐
                     │    │
    Dense Only       │ ───┼─────────── ⭐⭐⭐
                     │    │
    BM25 Only        │ ───┴─────────── ⭐⭐
                     │
                     └────────────────────────→ Komplexität
                     Easy        Medium     Hard

Chunking = Fundament (beeinflusst ALLE Stages!)
```

### Die Brücke zur Mathematik

**Intuition:** Jede Stage filtert und verbessert

**Mathematisch:**
```
Stage 1 (Chunking):
  Doc → [Chunk₁, Chunk₂, ..., Chunkₙ]
  n = Anzahl Chunks (abhängig von Strategie)

Stage 2 (Dense Retrieval):
  Score_dense(q, d) = cosine_sim(emb(q), emb(d))
  → Top-100 Kandidaten

Stage 3 (Sparse/BM25):
  Score_bm25(q, d) = Σ IDF(tᵢ) × saturation(tf(tᵢ, d))

Stage 4 (Hybrid Fusion):
  Score_hybrid = α × Score_dense + (1-α) × Score_bm25
  α ∈ [0, 1] tunable

Stage 5 (Cross-Encoder Re-Ranking):
  Score_rerank(q, d) = CrossEncoder([q, d])
  → Final Top-10
```

## 🧮 Das Konzept verstehen

### 1. Chunking: Das Fundament

#### Warum ist Chunking kritisch?

**Problem 1: Token-Limits**
```python
# BERT/Sentence-BERT: Max 512 Tokens
doc = """
Produktbeschreibung Laborkühlschrank LABO-288...
[2000 Tokens Text]
...Volumen: 280 Liter, Temperaturbereich: 2-8°C...
[weitere 1000 Tokens]
...Alarmfunktion bei Temperaturabweichung...
"""

# Was passiert ohne Chunking?
embedding = model.encode(doc[:512])  # ❌ Truncate!
# → "Alarmfunktion" ist NICHT im Embedding!
# → Query "Kühlschrank mit Alarm" findet das Doc NICHT!
```

**Problem 2: Embedding-Qualität**
```python
# Zu großer Chunk (viele Themen gemischt):
chunk_bad = """
Laborkühlschrank LABO-288 mit 280L.
[50 Sätze über Historie der Firma]
[30 Sätze über andere Produkte]
Alarmfunktion bei Abweichung.
"""

embedding_bad = model.encode(chunk_bad)
# → Embedding "verwässert" durch irrelevante Info
# → Similarity zu Query "Kühlschrank Alarm" niedriger!

# Guter Chunk (fokussiert auf ein Thema):
chunk_good = """
Laborkühlschrank LABO-288 Technische Spezifikationen:
- Volumen: 280 Liter
- Temperaturbereich: 2-8°C
- Alarmfunktion: Akustisch + optisch bei Abweichung
"""

embedding_good = model.encode(chunk_good)
# → Fokussiertes Embedding
# → Höhere Similarity zu Query!
```

#### Strategie 1: Fixed-Size Chunking

**Wie funktioniert es:**
```
Original: "Text Text Text Text Text Text Text Text"
          [512 tokens        ][512 tokens        ]
                    [Overlap  ]

Chunks:
  Chunk 1: Tokens 0-512
  Chunk 2: Tokens 462-974   ← 50 Token Overlap
  Chunk 3: Tokens 924-1436
```

**Warum Overlap?**
```python
# Ohne Overlap:
chunk_1 = "...Das Modell LABO-288"
chunk_2 = "hat 280 Liter Volumen..."
# → Zusammenhang zwischen Modell und Volumen verloren!

# Mit Overlap (50 Tokens):
chunk_1 = "...Das Modell LABO-288 hat 280..."
chunk_2 = "...LABO-288 hat 280 Liter Volumen..."
# → Beide Chunks enthalten Zusammenhang!
# → Query "LABO-288 Volumen" findet beide Chunks ✓
```

**Implementation:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # Max 512 tokens
    chunk_overlap=50,  # 10% Overlap
    separators=["\n\n", "\n", ". ", " ", ""],  # Versuche an natürlichen Grenzen zu splitten
    length_function=len,  # Oder token-basiert mit tiktoken
)

chunks = splitter.split_text(long_document)

# Beispiel-Output:
# Chunk 1: "Laborkühlschrank LABO-288\n\nTechnische Daten:\n..."
# Chunk 2: "...Volumen: 280 Liter\nTemperatur: 2-8°C\n..."
# → Splittet bevorzugt an Absätzen (\n\n), dann Sätzen (.)
```

**Wann nutzen:**
- Unstrukturierte Fließtexte (Artikel, Berichte)
- Wenn semantische Struktur nicht klar
- Default-Wahl wenn unsicher

**Trade-offs:**
- ✅ Einfach, vorhersagbar
- ✅ Funktioniert für meiste Texte
- ⚠️ Kann mitten im Satz splitten (trotz Separators)

---

#### Strategie 2: Semantic Chunking

**Wie funktioniert es:**
```
Markdown Doc:
  ## Section 1: Spezifikationen
  ...Text...

  ## Section 2: Alarmfunktion
  ...Text...

→ Chunks = Sections!
```

**Implementation:**
```python
import re

def semantic_chunk_markdown(markdown_text, min_chunk_size=100, max_chunk_size=800):
    # Split an Headings
    sections = re.split(r'\n#{1,3} ', markdown_text)

    chunks = []
    for section in sections:
        # Zu kurze Sections kombinieren
        if len(section) < min_chunk_size and chunks:
            chunks[-1] += '\n' + section
        # Zu lange Sections splitten (fallback zu Fixed-Size)
        elif len(section) > max_chunk_size:
            sub_chunks = split_fixed_size(section, max_chunk_size)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section)

    return chunks

# HTML:
from bs4 import BeautifulSoup

def semantic_chunk_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')

    chunks = []
    for section in soup.find_all(['section', 'article', 'div']):
        text = section.get_text()
        if 100 < len(text) < 800:
            chunks.append(text)

    return chunks
```

**Wann nutzen:**
- Strukturierte Dokumente (Markdown, HTML, PDF mit Sections)
- Technische Dokumentation
- Wikipedia-artige Texte

**Trade-offs:**
- ✅ Semantisch kohärent (ein Chunk = ein Thema)
- ✅ Natürliche Grenzen
- ⚠️ Variable Chunk-Größen (manche zu klein/groß)
- ⚠️ Erfordert Struktur im Dokument

---

#### Strategie 3: Sliding Window mit intelligentem Overlap

**Wie funktioniert es:**
```
Tokens: [t1, t2, t3, ..., t512, t513, ..., t1024]

Window 1: t1...t512
Window 2: t462...t974   ← Overlap = 50
Window 3: t924...t1436

→ Info an Chunk-Grenzen bleibt erhalten!
```

**Warum intelligenter als Simple Fixed-Size:**
```python
# Naive Fixed-Size:
# Chunk 1: "...Das Modell"
# Chunk 2: "LABO-288 hat..." ← "LABO-288" getrennt von "Modell"!

# Sliding Window mit Overlap:
# Chunk 1: "...Das Modell LABO-288 hat..."
# Chunk 2: "...Modell LABO-288 hat 280L..." ← Kontext erhalten!
```

**Wann nutzen:**
- Wenn Kontext-Erhaltung kritisch (Legal, Medical)
- Dense Texte ohne klare Struktur
- Als Fallback für Semantic Chunking

---

#### Chunk-Size Tuning

**Wie findet man optimale Chunk-Size?**

```python
# Experiment: Verschiedene Chunk-Sizes testen
chunk_sizes = [256, 512, 768, 1024]
overlaps = [0, 25, 50, 100]

results = {}
for size in chunk_sizes:
    for overlap in overlaps:
        chunks = chunk_document(doc, size, overlap)
        embeddings = model.encode(chunks)

        # Evaluate auf Test-Queries
        recall = evaluate_retrieval(test_queries, chunks, embeddings)
        results[(size, overlap)] = recall

# Best Configuration:
best = max(results.items(), key=lambda x: x[1])
print(f"Best: chunk_size={best[0][0]}, overlap={best[0][1]}, Recall={best[1]:.3f}")

# Typische Ergebnisse:
# Technische Docs: 512, overlap=50 → Recall=0.78
# Narrative Texte: 768, overlap=25 → Recall=0.75
# Code: variable (function-based) → Recall=0.82
```

**Empirische Guidelines:**
| Doc-Typ | Chunk-Size | Overlap | Warum? |
|---------|------------|---------|--------|
| **Technische Docs** | 300-500 | 10-20% | Viele kurze Fakten |
| **Narrative/Artikel** | 500-800 | 5-10% | Längere zusammenhängende Texte |
| **Legal/Medical** | 400-600 | 20-30% | Kontext kritisch |
| **Code** | Variable (Function) | 50-100 Zeilen | Semantische Einheiten |
| **Q&A Pairs** | 1 Pair = 1 Chunk | 0% | Bereits atomare Einheiten |

---

### 2. Re-Ranking: Von Ähnlichkeit zu Relevanz

#### Das Problem mit Bi-Encoder (Dense Retrieval)

**Warum reicht Dense nicht?**

```python
query = "Python Tutorial für absolute Anfänger"

# Bi-Encoder (Dense):
query_emb = model.encode(query)
doc_embs = model.encode(docs)
similarities = cosine_sim(query_emb, doc_embs)

# Top-3 Results:
# 1. "Python ist eine Programmiersprache"          Score: 0.76
# 2. "Programmieren lernen für Anfänger"           Score: 0.74
# 3. "Schritt-für-Schritt Python Tutorial"        Score: 0.72

# Problem: Doc #1 ist ähnlich, aber NICHT relevant!
#          Doc #3 ist am relevantesten, aber nur Platz 3!
```

**Warum passiert das?**

Bi-Encoder encoded Query und Doc **unabhängig**:
```
Query:    "Python Tutorial Anfänger"
            ↓ Encode
          [q_emb]

Doc:      "Python ist eine Programmiersprache"
            ↓ Encode
          [d_emb]

Similarity: cosine(q_emb, d_emb) = 0.76

→ Model sieht Query und Doc NIEMALS zusammen!
→ Kann nicht entscheiden: "Beantwortet Doc die Query?"
```

#### Cross-Encoder: Gemeinsames Verständnis

**Wie funktioniert Cross-Encoder anders?**

```
Input: "[CLS] Python Tutorial Anfänger [SEP] Python ist eine Programmiersprache [SEP]"
         ↓
       [BERT]
         ↓
     [CLS] Token
         ↓
  [Classification Head]
         ↓
   Relevance Score: 0.23  ← Niedrig! Doc beantwortet Query nicht.
```

Vs.

```
Input: "[CLS] Python Tutorial Anfänger [SEP] Schritt-für-Schritt Python Tutorial [SEP]"
         ↓
       [BERT]
         ↓
     [CLS] Token
         ↓
  [Classification Head]
         ↓
   Relevance Score: 0.94  ← Hoch! Doc beantwortet Query direkt.
```

**Cross-Attention:**
```
Query-Token "Tutorial" attended auf Doc-Token "Tutorial" → hohe Attention
Query-Token "Anfänger" attended auf Doc-Token "Schritt-für-Schritt" → versteht Kontext!

→ Model VERSTEHT den Zusammenhang zwischen Query und Doc!
```

#### Two-Stage Retrieval Pipeline

**Warum nicht nur Cross-Encoder?**

```python
# Naive: Cross-Encoder auf alle Docs
corpus_size = 1_000_000
for doc in corpus:
    score = cross_encoder.predict([query, doc])  # 10ms pro Pair

Total: 1M × 10ms = 10.000 Sekunden = 2.7 Stunden! ❌
```

**Lösung: Two-Stage**

```
Stage 1: Bi-Encoder (schnell, cached)
────────────────────────────────────────
1M Docs → Vector Search → Top-100 (100ms)

Stage 2: Cross-Encoder (präzise)
────────────────────────────────────────
100 Kandidaten → Cross-Encoder → Top-10 (200ms)

Total: 300ms ✓
```

**Implementation:**

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Setup
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Corpus embeddings (pre-computed, cached!)
corpus_embs = bi_encoder.encode(corpus, show_progress_bar=True)
# → Mache das EINMAL, speichere in Vector DB

# Query-Time:
def search_with_reranking(query, top_k=10):
    # Stage 1: Bi-Encoder Retrieval
    query_emb = bi_encoder.encode(query)
    hits = util.semantic_search(query_emb, corpus_embs, top_k=100)[0]

    # Stage 2: Cross-Encoder Re-Ranking
    candidates = [corpus[hit['corpus_id']] for hit in hits]
    pairs = [[query, doc] for doc in candidates]
    rerank_scores = cross_encoder.predict(pairs)

    # Sort by Cross-Encoder scores
    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return reranked[:top_k]

# Usage:
results = search_with_reranking("Python Tutorial für Anfänger")
for doc, score in results:
    print(f"{score:.4f}: {doc[:100]}...")
```

**Performance-Gewinn:**

Benchmark auf MS MARCO Dataset:
```
Bi-Encoder only:        MRR@10 = 0.33
+ Cross-Encoder (100):  MRR@10 = 0.39  (+18%!)
+ Cross-Encoder (1000): MRR@10 = 0.42  (+27%!)
```

---

### 3. Hybrid Search: Dense + Sparse vereint

#### Warum Hybrid?

**Dense allein:**
```python
query = "LABO-288 Kühlschrank 280 Liter"

# Dense versteht:
# "Kühlschrank" → "refrigerator", "Kühlaggregat", "Kühlgerät"
# ✓ Semantisch gut!

# Dense verliert:
# "LABO-288" → wird zu generischem "Model-ID" Vektor
# ❌ Exakte Modellnummer schlechter gewichtet!
```

**Sparse (BM25) allein:**
```python
# BM25 findet:
# "LABO-288" → exaktes Match! ✓
# "280" → exaktes Match! ✓

# BM25 verliert:
# "Kühlschrank" matched NICHT "refrigerator" ❌
# "Kühlaggregat" matched NICHT ❌
```

**Hybrid kombiniert:**
```
Dense:  Findet semantisch relevante Docs
BM25:   Boosted Docs mit exakten Keyword-Matches

→ Best of Both Worlds!
```

#### Score Fusion: Wie kombiniert man?

**Method 1: Weighted Sum (am häufigsten)**

```python
Score_hybrid = α × Score_dense + (1-α) × Score_bm25
```

**Problem:** Verschiedene Skalen!
```python
Score_dense: 0.0 - 1.0  (Cosine Similarity)
Score_bm25:  0.0 - 150.0 (kann sehr hoch werden!)

→ Naives α=0.5 gibt BM25 zu viel Gewicht!
```

**Lösung: Min-Max Normalization**

```python
def normalize(scores):
    min_s = scores.min()
    max_s = scores.max()
    return (scores - min_s) / (max_s - min_s + 1e-9)

# Normalize beide zu [0, 1]
dense_norm = normalize(dense_scores)
bm25_norm = normalize(bm25_scores)

# Jetzt fair kombinieren:
alpha = 0.5
hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
```

**Alpha-Tuning:**

```python
# Experiment: Finde bestes α
alphas = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0

for alpha in alphas:
    hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
    recall = evaluate(hybrid_scores, ground_truth)
    print(f"α={alpha:.1f}: Recall@10={recall:.3f}")

# Typische Ergebnisse:
# α=0.0 (nur BM25):   Recall=0.62
# α=0.3:              Recall=0.74  ← BM25-heavy
# α=0.5:              Recall=0.78  ← Balanced
# α=0.7:              Recall=0.81  ← Dense-heavy ✓
# α=1.0 (nur Dense):  Recall=0.75
```

**Guidelines für α:**

| Use Case | α-Wert | Warum? |
|----------|--------|--------|
| **E-Commerce (Produktsuche)** | 0.3-0.4 | Exakte SKUs, Modellnummern wichtig |
| **Q&A / Semantic Search** | 0.6-0.8 | Semantik wichtiger als Keywords |
| **Legal / Medical** | 0.5 | Balance: Fachbegriffe + Semantik |
| **Code Search** | 0.3-0.5 | Funktionsnamen (exakt) + Bedeutung |

---

**Method 2: Reciprocal Rank Fusion (RRF)**

```python
# Idee: Ranglisten statt Scores kombinieren

# Dense Ranking:
# Doc A: Rang 1
# Doc B: Rang 5
# Doc C: Rang 3

# BM25 Ranking:
# Doc A: Rang 10
# Doc B: Rang 1
# Doc C: Rang 2

# RRF Score:
def rrf_score(rank, k=60):
    return 1 / (k + rank)

# Doc A: 1/(60+1) + 1/(60+10) = 0.0307
# Doc B: 1/(60+5) + 1/(60+1)  = 0.0318  ← Highest!
# Doc C: 1/(60+3) + 1/(60+2)  = 0.0320  ← Highest!

→ RRF ist robuster gegen Score-Skalen!
```

**Wann RRF statt Weighted Sum?**
- Wenn Scores schwer zu normalisieren (z.B. Neural Reranker + BM25)
- Als Baseline (funktioniert oft gut ohne Tuning)
- Weighted Sum meist besser mit Tuning, aber RRF ist robuster

---

## ⚠️ Häufige Missverständnisse

### ❌ Missverständnis 1: "Größere Chunks = bessere Embeddings"

**Warum falsch:**

```python
# Zu großer Chunk (800 tokens):
chunk_large = """
Liebherr Geschichte seit 1949...
[200 Tokens über Historie]
...LABO-288 Kühlschrank...
[50 Tokens relevante Info]
...andere Produkte...
[550 Tokens irrelevant]
"""

embedding_large = model.encode(chunk_large)
# → "Liebherr", "Geschichte", "1949" dominieren Embedding
# → "LABO-288", "Kühlschrank" verwässert!

# Query: "LABO-288 Spezifikationen"
# → Niedrige Similarity! (zu viel Noise)
```

**✓ Richtig:**
```python
# Fokussierter Chunk (300 tokens):
chunk_good = """
LABO-288 Laborkühlschrank Spezifikationen:
Volumen: 280 Liter
Temperatur: 2-8°C
Alarmfunktion: Ja
"""

embedding_good = model.encode(chunk_good)
# → Fokussiert auf relevante Info
# → Höhere Similarity zu Query! ✓
```

**Merksatz:** "Ein Chunk = Ein Thema. Mehr Info ≠ Bessere Embeddings!"

### ❌ Missverständnis 2: "Cross-Encoder für alle Docs nutzen"

**Warum falsch:**

```python
# 1 Million Docs × 10ms Cross-Encoder = 2.7 Stunden!
# User wartet nicht so lange ❌
```

**✓ Richtig:**

Two-Stage: Bi-Encoder (1M → 100) + Cross-Encoder (100 → 10) = 300ms ✓

**Merksatz:** "Cross-Encoder nur für Re-Ranking, nicht für Initial Retrieval!"

### ❌ Missverständnis 3: "α=0.5 ist immer optimal für Hybrid"

**Warum falsch:**

```python
# E-Commerce (viele Modellnummern):
query = "iPhone 15 Pro Max 256GB"
# α=0.5: Recall@10 = 0.68
# α=0.3: Recall@10 = 0.81  ← BM25 wichtiger! ✓

# Q&A (semantisch):
query = "Wie funktioniert Quantenphysik?"
# α=0.5: Recall@10 = 0.72
# α=0.7: Recall@10 = 0.84  ← Dense wichtiger! ✓
```

**✓ Richtig:**

Benchmark auf eigenem Dataset, tune α!

**Merksatz:** "α hängt vom Use Case ab - immer benchmarken!"

---

## 🔬 Hands-On: Complete Optimized Pipeline

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

# ═══════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

long_documents = [...]  # Your corpus

# ═══════════════════════════════════════════════════════════
# Stage 1: Chunking
# ═══════════════════════════════════════════════════════════
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = []
chunk_to_doc = {}  # Track which chunk belongs to which doc

for doc_id, doc in enumerate(long_documents):
    doc_chunks = splitter.split_text(doc)
    for chunk in doc_chunks:
        chunk_id = len(chunks)
        chunks.append(chunk)
        chunk_to_doc[chunk_id] = doc_id

print(f"Created {len(chunks)} chunks from {len(long_documents)} docs")

# ═══════════════════════════════════════════════════════════
# Stage 2: Embed (Pre-compute, cache!)
# ═══════════════════════════════════════════════════════════
chunk_embs = bi_encoder.encode(
    chunks,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Normalize for faster dot product
chunk_embs_norm = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True)

# ═══════════════════════════════════════════════════════════
# Stage 3: BM25 Index
# ═══════════════════════════════════════════════════════════
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# ═══════════════════════════════════════════════════════════
# Query Function: Hybrid + Re-Ranking
# ═══════════════════════════════════════════════════════════
def search_optimized(query, alpha=0.6, top_k=10):
    # 1. Dense Retrieval
    query_emb = bi_encoder.encode(query, convert_to_numpy=True)
    query_emb_norm = query_emb / np.linalg.norm(query_emb)
    dense_scores = np.dot(chunk_embs_norm, query_emb_norm)

    # 2. BM25
    bm25_scores = bm25.get_scores(query.lower().split())

    # 3. Normalize
    dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

    # 4. Hybrid Fusion
    hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

    # 5. Top-100 Kandidaten
    top_100_indices = np.argsort(hybrid_scores)[-100:][::-1]

    # 6. Cross-Encoder Re-Ranking
    candidates = [chunks[i] for i in top_100_indices]
    pairs = [[query, doc] for doc in candidates]
    rerank_scores = cross_encoder.predict(pairs)

    # 7. Final Ranking
    final_results = []
    for idx, score in zip(top_100_indices, rerank_scores):
        final_results.append({
            'chunk_id': idx,
            'doc_id': chunk_to_doc[idx],
            'text': chunks[idx],
            'score': score
        })

    # Sort by rerank score
    final_results.sort(key=lambda x: x['score'], reverse=True)

    return final_results[:top_k]

# ═══════════════════════════════════════════════════════════
# Usage
# ═══════════════════════════════════════════════════════════
query = "Laborkühlschrank 280 Liter mit Temperaturalarm"
results = search_optimized(query, alpha=0.6)

print(f"\nQuery: {query}\n")
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.4f}")
    print(f"   Doc ID: {result['doc_id']}")
    print(f"   Text: {result['text'][:150]}...\n")
```

**Was beobachten:**
- Chunking: Wie viele Chunks pro Doc?
- Hybrid: Teste verschiedene α-Werte (0.3, 0.5, 0.7)
- Re-Ranking: Scores ändern sich signifikant nach Cross-Encoder
- Performance: Messe Latenz für jeden Stage

---

## 📊 Performance Benchmarks

### Recall@10 Improvements

```
Baseline (Dense only, no chunking):     0.60
+ Optimized Chunking:                   0.66  (+10%)
+ Hybrid (Dense + BM25):                0.75  (+25%)
+ Cross-Encoder Re-Ranking:             0.85  (+42%!)
```

### Latency Breakdown

```
Chunking:           0ms    (pre-computed)
Dense Retrieval:    50ms   (vector search)
BM25:               20ms   (inverted index)
Hybrid Fusion:      5ms    (score combination)
Cross-Encoder:      200ms  (100 pairs)
──────────────────────────────────────────
Total:              275ms  ✓ (acceptable!)
```

### Trade-off Matrix

| Configuration | Recall@10 | Latency | Complexity | When? |
|---------------|-----------|---------|------------|-------|
| **Dense only** | 0.60 | 50ms | Low | Prototyping |
| **Dense + BM25** | 0.75 | 70ms | Medium | Production baseline |
| **+ Re-Ranking (Top-100)** | 0.85 | 270ms | High | High-quality search |
| **+ Re-Ranking (Top-1000)** | 0.87 | 2s | Very High | Offline/Batch |

## 🚀 Was du jetzt kannst

**Chunking-Expertise:**
- ✓ Du verstehst WARUM Chunking kritisch ist (Token-Limits + Embedding-Qualität)
- ✓ Du wählst Strategie für Doc-Typ (Fixed-Size vs. Semantic vs. Sliding Window)
- ✓ Du tunest Chunk-Size und Overlap basierend auf Evaluation

**Re-Ranking-Expertise:**
- ✓ Du verstehst Unterschied Bi-Encoder vs. Cross-Encoder (Ähnlichkeit vs. Relevanz)
- ✓ Du implementierst Two-Stage Retrieval (schnell + präzise)
- ✓ Du gewinnst +10-20% Recall durch Re-Ranking

**Hybrid Search:**
- ✓ Du kombinierst Dense + BM25 mit Weighted Sum
- ✓ Du normalisierst Scores korrekt (Min-Max)
- ✓ Du tunest α-Parameter für deinen Use Case

**Production-Ready:**
- ✓ Du baust komplette optimierte Pipeline
- ✓ Du benchmarkst jede Optimization (messbare Improvements)
- ✓ Du verstehst Latenz-Trade-offs

## 🔗 Weiterführende Themen

**Nächster Schritt:**
→ [05-production-considerations.md](05-production-considerations.md) - Vector DBs, Quantization, Scaling

**Vertiefung:**
→ [../infrastructure/vector-databases.md](../infrastructure/vector-databases.md) - Deep Dive Indexing

**Praktisch:**
→ [../../06-applications/01-rag-systems.md](../../06-applications/01-rag-systems.md) - Complete RAG Implementation
