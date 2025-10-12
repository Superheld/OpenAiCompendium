# Embedding Architectures: Wie lernen Maschinen Bedeutung?

## ❓ Das Problem (Problem-First)

**Ohne Embedding-Architekturen geht folgendes schief:**
- **Nur Keyword-Matching**: "Kühlschrank" findet nicht "Kühlaggregat" oder "refrigerator" - semantisches Verständnis fehlt
- **Bag-of-Words verliert Kontext**: "Bank am Fluss" vs. "Bank hat geschlossen" - gleiches Wort, völlig andere Bedeutung
- **Skalierung unmöglich**: Dense findet Synonyme aber verliert exakte Matches, BM25 findet Keywords aber keine Semantik - du musst zwischen beidem wählen

**Die zentrale Frage:**
Welche Embedding-Architektur nutze ich wann? Dense, Sparse, Multi-Vector oder Cross-Encoder - und warum gibt es überhaupt so viele verschiedene Ansätze?

**Beispiel-Szenario:**
```python
Query: "Kühlgerät für Arzneimittel mit Temperaturüberwachung"

# BM25 (Sparse): Findet nur exakte Wörter
Doc 1: "Kühlgerät Arzneimittel Temperaturüberwachung" → Score: 5.2 ✓
Doc 2: "Medikamentenkühlschrank mit Alarm-System"     → Score: 0.0 ✗

# Dense: Versteht Semantik, verliert aber Präzision
Doc 1: "Kühlgerät Arzneimittel Temperaturüberwachung" → Score: 0.82 ✓
Doc 2: "Medikamentenkühlschrank mit Alarm-System"     → Score: 0.79 ✓
Doc 3: "Labor-Equipment Kühlung"                       → Score: 0.75 ❓ (zu fuzzy!)

# Was wir wirklich brauchen: Beides!
```

## 🎯 Lernziele

Nach diesem Kapitel kannst du:
- [ ] Du verstehst die 4 Haupt-Architekturen (Dense, Sparse, Multi-Vector, Cross-Encoder) und ihre Trade-offs
- [ ] Du kannst begründet entscheiden welche Architektur für deinen Use Case passt
- [ ] Du implementierst alle 4 Architekturen praktisch und verstehst wann Hybrid-Ansätze nötig sind

## 🧠 Intuition zuerst (Scaffolded Progression)

### Alltagsanalogie: Verschiedene Arten zu suchen

**Beispiel aus dem echten Leben: Buch in einer Bibliothek finden**

**1. Index-Suche (= Sparse / BM25):**
```
Du suchst: "Quantenphysik"
→ Schaust im Index: "Quantenphysik" → Seite 142, 287, 453
→ Sehr schnell, aber nur EXAKTE Begriffe
→ Findet NICHT: "Quantenmechanik", "Teilchenphysik"
```

**2. Inhaltliche Suche (= Dense Embeddings):**
```
Du fragst Bibliothekar: "Etwas über Quantenphysik"
→ Bibliothekar versteht: "Ah, du meinst moderne Physik, Teilchen..."
→ Zeigt dir auch: Quantenmechanik, Heisenberg, Schrödinger
→ Langsamer, aber VERSTEHT was du meinst
```

**3. Wort-für-Wort Vergleich (= Multi-Vector / ColBERT):**
```
Du vergleichst jedes Wort deiner Notiz mit jedem Wort im Buch
→ "Quanten" matched "Quantenmechanik" ✓
→ "Physik" matched "Physik" ✓
→ Sehr präzise, aber aufwändig
```

**4. Bibliothekar liest beide (= Cross-Encoder):**
```
Bibliothekar liest deine Frage UND das Buch zusammen
→ Perfektes Verständnis ob es passt
→ Aber: Muss JEDES Buch lesen (sehr langsam!)
```

### Visualisierung: Die 4 Architekturen

```
                    Geschwindigkeit
                         ↑
    Sparse (BM25)        │      Dense (Sentence-BERT)
         ⚡⚡⚡            │            ⚡⚡
    Exakte Keywords      │       Semantisch
    Keine Semantik       │       Fuzzy Matches
                         │
    ─────────────────────┼──────────────────────→ Genauigkeit
                         │
    Multi-Vector         │      Cross-Encoder
    (ColBERT)            │        (Re-Ranker)
         ⚡              │             ⚡
    Token-Level          │       Maximale
    Präzision            │       Genauigkeit
                         ↓
                   Rechenaufwand
```

**Trade-off verstehen:**
- **Links-Oben (Sparse)**: Schnell, aber nur Keywords
- **Rechts-Oben (Dense)**: Balance von Speed + Semantik
- **Links-Unten (Multi-Vector)**: Präzise, aber langsamer
- **Rechts-Unten (Cross-Encoder)**: Perfekt, aber nur für Re-Ranking

### Die Brücke zur Mathematik

**Intuition:** Verschiedene Wege Text in Zahlen zu verwandeln

**Mathematisch:**
- **Sparse**: $v \in \mathbb{R}^{|V|}$ wo 99% = 0 (nur Wörter die vorkommen)
- **Dense**: $v \in \mathbb{R}^d$ wo alle Werte $\neq 0$ (d = 384-1024)
- **Multi-Vector**: $M \in \mathbb{R}^{n \times d}$ (n Token-Vektoren statt einem)
- **Cross-Encoder**: $f(query, doc) \rightarrow [0, 1]$ (keine Vektoren!)

## 🧮 Das Konzept verstehen

### 1. Dense Embeddings (Sentence-BERT)

#### Wie funktioniert es?

**Architektur:**
```
Text: "Laborkühlschrank 280L"
  ↓
[Tokenization]
  ["Labor", "##kühl", "##schrank", "280", "L"]
  ↓
[BERT Transformer × 12 Layer]
  Self-Attention lernt Kontext
  ↓
Token-Embeddings: [5, 768]
  ↓
[Mean Pooling]
  Durchschnitt über alle Token
  ↓
Sentence-Embedding: [768]
  [0.234, -0.456, 0.123, ..., -0.234]
```

**Intuition hinter der Formel:**

Mean Pooling:
$$\text{emb}_{\text{sentence}} = \frac{1}{n} \sum_{i=1}^{n} \text{emb}_{\text{token}_i}$$

**Warum Mean Pooling?**
- Alle Token tragen bei
- Dokumentlänge spielt keine Rolle (normalisiert)
- Robust und funktioniert praktisch immer

**Schritt-für-Schritt:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 1. Text → Tokens
text = "Laborkühlschrank"
# Intern: ["Labor", "##kühl", "##schrank"]

# 2. Transformer Processing
# Self-Attention: Jedes Token schaut auf alle anderen
# "##schrank" lernt von "Labor" und "##kühl"

# 3. Pooling
embedding = model.encode(text)  # [384]

# 4. Similarity
query_emb = model.encode("Medikamentenkühlschrank")
from sentence_transformers import util
similarity = util.cos_sim(embedding, query_emb)
# → 0.87 (sehr ähnlich!)
```

#### Training: Contrastive Learning

**Prinzip:** Ähnliche Texte → ähnliche Vektoren

```
Positive Pair:
  "Laborkühlschrank 280L"
  "Medizinischer Kühlschrank 280 Liter"
  → Embeddings sollen NAH sein

Negative Pair:
  "Laborkühlschrank 280L"
  "Pommes Frites"
  → Embeddings sollen WEIT sein
```

**Loss-Funktion (Multiple Negatives Ranking):**

$$\mathcal{L} = -\log \frac{e^{\text{sim}(q, p^+)}}{\sum_{p \in \{p^+, p^-_1, ..., p^-_k\}} e^{\text{sim}(q, p)}}$$

**In Worten:**
- Maximiere Similarity zum positiven Beispiel
- Minimiere Similarity zu allen negativen Beispielen im Batch

#### Varianten & Trade-offs

| Model | Dimensionen | Speed | Qualität | Use-Case |
|-------|-------------|-------|----------|----------|
| **all-MiniLM-L6-v2** | 384 | ⚡⚡⚡ | ⭐⭐⭐ | Schnelles Prototyping |
| **all-mpnet-base-v2** | 768 | ⚡⚡ | ⭐⭐⭐⭐ | Production (Englisch) |
| **multilingual-e5-large** | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | Multilinguale Production |

---

### 2. Sparse Embeddings (BM25 & SPLADE)

#### Klassisch: BM25

**Intuition:** Nur Wörter die vorkommen haben Gewichte

```python
Text: "Laborkühlschrank 280L"

# Sparse Vector (30.000 Dimensionen)
{
    5421: 3.2,   # "Laborkühlschrank"
    8934: 1.5,   # "280L"
    # Alle anderen 29.998 Dimensionen: 0
}
```

**BM25 Formel:**

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

**Intuition hinter der Formel:**
- **IDF**: Seltene Wörter = wichtiger
- **Saturation ($k_1$)**: Nach 2-3 Vorkommen zählt mehr nicht viel
- **Length Norm ($b$)**: Kurze Docs werden bevorzugt

**Warum dieser Ansatz?**
- TF-IDF wächst linear (10× Wort = 10× Score) → unrealistisch
- BM25 saturiert (10× Wort ≈ 2× Score) → realistischer

#### Modern: SPLADE (Learned Sparse)

**Das Beste aus beiden Welten:**

```python
Input: "Kühlgerät"

# BM25 Output (nur Input):
{
    "Kühlgerät": 3.5
}

# SPLADE Output (+ Expansions!):
{
    "Kühlgerät": 3.5,      # Original
    "Kühlschrank": 2.1,    # Gelernt!
    "Kühlanlage": 1.8,     # Synonym
    "refrigerator": 0.9,   # Cross-lingual!
}
```

**Architektur:**

```
Text: "Kühlgerät"
  ↓
[BERT Transformer]
  ↓
[MLM Head]  ← Vorhersage für JEDES Wort im Vokabular
  [30.000] logits
  ↓
[ReLU + Log(1+x)]  ← Sparsity
  ↓
Sparse Vector: nur ~50 non-zero
```

**Training:** Contrastive + Sparsity Regularization

$$\mathcal{L} = \mathcal{L}_{\text{ranking}} + \lambda \cdot \|v\|_1$$

- $\mathcal{L}_{\text{ranking}}$: Positive Docs ranken höher
- $\lambda \cdot \|v\|_1$: Strafe für zu viele non-zero Werte

#### Varianten & Trade-offs

| Methode | Semantik | Speed | Speicher | Use-Case |
|---------|----------|-------|----------|----------|
| **BM25** | ❌ | ⚡⚡⚡ | ✅✅✅ | Keyword-Search, Baseline |
| **SPLADE** | ✅ | ⚡⚡ | ✅✅ | Production Hybrid, Best of Both |

---

### 3. Multi-Vector Embeddings (ColBERT)

#### Das Problem mit Single-Vector Dense

**Informationsverlust durch Pooling:**

```python
Text: "Laborkühlschrank mit Temperaturüberwachung"

# Dense (Single Vector):
embedding = mean([emb_Labor, emb_kühlschrank, emb_Temperatur, ...])
# → Ein 768-dim Vektor
# → Details gehen verloren!

# Multi-Vector (ColBERT):
embeddings = [emb_Labor, emb_kühlschrank, emb_Temperatur, ...]
# → 5× 128-dim Vektoren
# → Jedes Wort behält eigene Nuance!
```

#### ColBERT Architecture

**Late Interaction:**

```
Query: "Kühlschrank Alarm"
  ↓ BERT
[emb_Kühlschrank, emb_Alarm]  ← Query Embeddings

Document: "Medikamentenkühlschrank mit Alarm-System"
  ↓ BERT
[emb_Medikamenten, emb_kühlschrank, emb_Alarm, emb_System]  ← Doc Embeddings

─────────────────────────────────────────────────────
Late Interaction: MaxSim

Für jedes Query-Token → finde bestes Doc-Token:

  "Kühlschrank" matched am besten mit "kühlschrank" → 0.95
  "Alarm"       matched am besten mit "Alarm"       → 0.98

Score = 0.95 + 0.98 = 1.93
```

**MaxSim Formel:**

$$\text{Score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \text{sim}(q_i, d_j)$$

**Intuition:**
- Jedes Query-Wort sucht bestes Match im Document
- Summe aller besten Matches = Gesamt-Score
- **Präziser** als Single-Vector (keine Info geht verloren)

**Warum dieser Ansatz?**
- **Cross-Encoder:** $O(n)$ - muss jedes Doc neu encoden
- **Single-Vector:** $O(1)$ - aber Informationsverlust
- **ColBERT:** $O(k)$ - k = avg tokens, aber behält Präzision!

#### Code-Beispiel

```python
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig

# Setup
with Run().context(RunConfig(nranks=1)):
    # Index erstellen
    indexer = Indexer(checkpoint='colbert-ir/colbertv2.0')
    indexer.index(
        name='my_index',
        collection=['Doc 1 text', 'Doc 2 text', ...],
        overwrite=True
    )

    # Suche
    searcher = Searcher(index='my_index', checkpoint='colbert-ir/colbertv2.0')
    results = searcher.search("Laborkühlschrank mit Alarm", k=10)

    for doc_id, rank, score in results:
        print(f"#{rank} Doc {doc_id}: {score:.2f}")
```

#### Varianten & Trade-offs

| Aspekt | Single-Vector | Multi-Vector (ColBERT) |
|--------|---------------|------------------------|
| **Speicher** | 768 floats/doc | 768 × avg_tokens floats/doc (~10x mehr) |
| **Geschwindigkeit** | ⚡⚡⚡ | ⚡⚡ |
| **Genauigkeit** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Use-Case** | Standard RAG | High-Precision Retrieval |

---

### 4. Cross-Encoders (Re-Ranking)

#### Warum Cross-Encoders?

**Bi-Encoder (Dense) Problem:**

```python
# Bi-Encoder: Query und Doc UNABHÄNGIG encoded
query_emb = model.encode("Kühlschrank Alarm")
doc_emb = model.encode("Medikamenten-Kühlschrank mit Temperatur-Alarm")

# Similarity:
score = cosine_sim(query_emb, doc_emb)  # 0.78

# Problem: Model sieht Query und Doc NIE zusammen!
# → Kann Nuancen nicht verstehen
```

**Cross-Encoder: Query + Doc ZUSAMMEN:**

```python
# Cross-Encoder: Concatenate Query + Doc
input_text = "[CLS] Kühlschrank Alarm [SEP] Medikamenten-Kühlschrank Alarm [SEP]"
  ↓ BERT
  ↓ Classification Head
score = 0.94  # Höher! Model versteht Kontext besser
```

#### Architektur

```
Query: "Kühlschrank Alarm"
Document: "Medikamenten-Kühlschrank mit Alarm-System"

┌───────────────────────────────────────────────────┐
│  [CLS] Query [SEP] Document [SEP]                 │
│  ↓                                                 │
│  [BERT Transformer × 12]                          │
│    Self-Attention über Query UND Doc!             │
│  ↓                                                 │
│  [CLS] Token Embedding                            │
│  ↓                                                 │
│  [Linear Layer + Sigmoid]                         │
│  ↓                                                 │
│  Relevance Score: 0.94                            │
└───────────────────────────────────────────────────┘
```

**Warum besser?**
- **Bi-Encoder:** $\text{emb}(q) \cdot \text{emb}(d)$ - unabhängig
- **Cross-Encoder:** $f(q, d)$ - Query und Doc interagieren via Attention!

**Training:**

```python
# Positive Pairs:
("[CLS] Query [SEP] Relevant Doc [SEP]", label=1)

# Negative Pairs:
("[CLS] Query [SEP] Irrelevant Doc [SEP]", label=0)

# Loss: Binary Cross-Entropy
loss = BCE(predicted_score, true_label)
```

#### Re-Ranking Pipeline

**Zwei-Stufen Retrieval:**

```
Stage 1: Bi-Encoder (schnell, 1000 Kandidaten)
────────────────────────────────────────────────────
Query → Bi-Encoder → Vector DB Search
        → Top-1000 Dokumente (100ms)

Stage 2: Cross-Encoder (präzise, Top-10)
────────────────────────────────────────────────────
Query + Top-1000 Docs → Cross-Encoder Re-Ranking
        → Top-10 präzise geranked (500ms)
```

**Warum nicht nur Cross-Encoder?**
- 1M Docs × Cross-Encoder = **zu langsam** (10+ Sekunden)
- Bi-Encoder → 1000 Kandidaten (schnell)
- Cross-Encoder → 1000 → 10 (präzise)

#### Code-Beispiel

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Stage 1: Bi-Encoder Retrieval
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
corpus = ["Doc 1", "Doc 2", ..., "Doc 1000000"]
corpus_embs = bi_encoder.encode(corpus)

query = "Laborkühlschrank mit Alarm"
query_emb = bi_encoder.encode(query)

# Top-100 via Cosine Similarity
hits = util.semantic_search(query_emb, corpus_embs, top_k=100)[0]
candidates = [corpus[hit['corpus_id']] for hit in hits]

# Stage 2: Cross-Encoder Re-Ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc] for doc in candidates]
scores = cross_encoder.predict(pairs)

# Re-Rank nach Cross-Encoder Scores
reranked_hits = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

print("Top-10 nach Re-Ranking:")
for doc, score in reranked_hits[:10]:
    print(f"{score:.4f}: {doc}")
```

#### Varianten & Trade-offs

| Aspekt | Bi-Encoder | Cross-Encoder |
|--------|------------|---------------|
| **Speed** | ⚡⚡⚡ (cached embeddings) | ⚡ (muss jeden Pair encoden) |
| **Genauigkeit** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Skalierung** | Millionen Docs ✓ | Nur für Top-K Re-Ranking |
| **Use-Case** | Retrieval | Re-Ranking |

---

## ⚠️ Häufige Missverständnisse (Misconception Debugging)

### ❌ Missverständnis 1: "Dense ist immer besser als Sparse"

**Warum das falsch ist:**

```python
Query: "LABO-288-PRO"  # Modellnummer

# Dense (Semantic):
# Findet: "Laborkühlschrank", "Labor-Equipment", ...
# → Zu fuzzy! User will EXAKT "LABO-288-PRO"

# BM25 (Sparse):
# Findet: Dokument mit "LABO-288-PRO"
# → Perfekt!
```

**✓ Richtig ist:**
- **Exakte Matches** (IDs, SKUs, Modellnummern) → Sparse
- **Semantische Suche** (Synonyme, Paraphrasen) → Dense
- **Production** → Hybrid (beides!)

**Merksatz:**
"Sparse für Keywords, Dense für Semantik, Hybrid für Production!"

### ❌ Missverständnis 2: "Cross-Encoder für alles nutzen"

**Warum das falsch ist:**

```python
# 1 Million Dokumente
# Cross-Encoder muss JEDEN Query-Doc-Pair encoden
# = 1.000.000 Forward-Passes
# = 10+ Sekunden pro Query ❌

# Bi-Encoder:
# Embeddings cached
# Vector Search: 100ms ✓
```

**✓ Richtig ist:**
Cross-Encoder **nur für Re-Ranking** der Top-K Kandidaten!

**Merksatz:**
"Bi-Encoder holt Kandidaten, Cross-Encoder findet den Besten!"

### ❌ Missverständnis 3: "ColBERT ist immer besser als Single-Vector"

**Warum das falsch ist:**

```python
# ColBERT Speicher:
1M Docs × 50 avg tokens × 128 dims × 4 bytes
= 25 GB

# Single-Vector:
1M Docs × 384 dims × 4 bytes
= 1.5 GB

# 16x mehr Speicher!
```

**✓ Richtig ist:**
- **Kleiner Corpus (<100k)** → ColBERT (Präzision wichtiger)
- **Großer Corpus (>1M)** → Single-Vector (Speicher wichtiger)
- **High-Value Queries** (z.B. Legal, Medical) → ColBERT

**Merksatz:**
"ColBERT für Qualität, Single-Vector für Skalierung!"

## 🔬 Hands-On: Alle 4 Architekturen vergleichen

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
import numpy as np

# Corpus
corpus = [
    "Laborkühlschrank mit 280 Liter Volumen und Temperaturüberwachung",
    "Medikamentenkühlschrank nach DIN 13277 mit Alarm-System",
    "Gefrierschrank für Labor mit -40°C Temperatur",
    "Pommes Frites Zubereitung in der Fritteuse"
]

query = "Kühlschrank für Medikamente mit Alarm"

print(f"Query: {query}\n")

# ═══════════════════════════════════════════════════════════
# 1. BM25 (Sparse)
# ═══════════════════════════════════════════════════════════
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(query.lower().split())

print("1. BM25 (Sparse):")
for idx, score in enumerate(bm25_scores):
    print(f"  {score:.4f}: {corpus[idx][:60]}...")

# ═══════════════════════════════════════════════════════════
# 2. Dense (Bi-Encoder)
# ═══════════════════════════════════════════════════════════
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
query_emb = bi_encoder.encode(query, convert_to_tensor=True)
corpus_embs = bi_encoder.encode(corpus, convert_to_tensor=True)
dense_scores = util.cos_sim(query_emb, corpus_embs)[0].cpu().numpy()

print("\n2. Dense (Bi-Encoder):")
for idx, score in enumerate(dense_scores):
    print(f"  {score:.4f}: {corpus[idx][:60]}...")

# ═══════════════════════════════════════════════════════════
# 3. Hybrid (BM25 + Dense)
# ═══════════════════════════════════════════════════════════
# Normalize scores to [0, 1]
bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
dense_norm = dense_scores

# Weighted combination
alpha = 0.5  # 50% BM25, 50% Dense
hybrid_scores = alpha * bm25_norm + (1 - alpha) * dense_norm

print("\n3. Hybrid (BM25 + Dense):")
for idx, score in enumerate(hybrid_scores):
    print(f"  {score:.4f}: {corpus[idx][:60]}...")

# ═══════════════════════════════════════════════════════════
# 4. Cross-Encoder (Re-Ranking Top-3 from Dense)
# ═══════════════════════════════════════════════════════════
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Top-3 Kandidaten from Dense
top_3_indices = np.argsort(dense_scores)[-3:][::-1]
top_3_docs = [corpus[idx] for idx in top_3_indices]

# Cross-Encoder Re-Ranking
pairs = [[query, doc] for doc in top_3_docs]
cross_scores = cross_encoder.predict(pairs)

print("\n4. Cross-Encoder (Re-Ranking Top-3):")
for idx, score in zip(top_3_indices, cross_scores):
    print(f"  {score:.4f}: {corpus[idx][:60]}...")
```

**Was du beobachten solltest:**
- **BM25** rankt Doc mit exaktem "Medikamente" + "Alarm" hoch
- **Dense** findet auch "Medikamentenkühlschrank" (Synonym!)
- **Hybrid** kombiniert beste von beiden
- **Cross-Encoder** gibt präziseste Scores für Top-Kandidaten

**Experimentiere selbst:**
- Was passiert wenn Query = "LABO-288" (Modellnummer)? Welche Architektur gewinnt?
- Wie ändert sich `alpha` im Hybrid? Teste 0.2, 0.5, 0.8
- Was wenn du Cross-Encoder auf ALLE Docs anwendest? (Zeit messen!)

## ⏱️ 5-Minuten-Experte

### 1. Verständnisfrage: Warum ist Mean Pooling der Standard?

<details><summary>💡 Zeige Antwort</summary>

**Antwort:**
Mean Pooling ist längenunabhängig und nutzt alle Token-Informationen.

**Erklärung:**
- **CLS Pooling:** Nur ein Token - verschwendet Info
- **Max Pooling:** Nur stärkste Signale - kann "spiky" sein
- **Mean Pooling:** Durchschnitt aller Token - robust und fair für kurze/lange Texte

**Merksatz:**
"Mean ist der demokratische Durchschnitt - jedes Token zählt gleich!"

</details>

### 2. Anwendungsfrage: Dein RAG System ist zu langsam - was tun?

<details><summary>💡 Zeige Antwort</summary>

**Antwort:**
Zwei-Stufen Retrieval: Dense Bi-Encoder (Top-100) → Cross-Encoder Re-Ranking (Top-10)

**Begründung:**
```python
# Vorher: Cross-Encoder auf alles
1M Docs × Cross-Encoder = 10+ Sekunden ❌

# Nachher: Two-Stage
Stage 1: Bi-Encoder → Top-100 (100ms)
Stage 2: Cross-Encoder → Top-10 (200ms)
Total: 300ms ✓
```

**Alternative:**
- ColBERT statt Cross-Encoder (etwas schneller, cached embeddings)
- Quantization (int8 statt float32)
- Kleineres Model (MiniLM statt MPNet)

</details>

### 3. Trade-off-Frage: Wann ColBERT statt Single-Vector Dense?

<details><summary>💡 Zeige Antwort</summary>

**Antwort:**
Kommt auf Corpus-Größe und Qualitäts-Anforderungen an.

**Kontext matters:**

| Szenario | Wahl | Warum? |
|----------|------|--------|
| <100k Docs, High-Value (Legal, Medical) | ColBERT | Präzision wichtiger, Speicher egal |
| >1M Docs, Consumer App | Single-Vector | Speicher & Latenz wichtiger |
| Scientific Papers, Code Search | ColBERT | Token-Level Matching kritisch |
| E-Commerce, News | Single-Vector | Skalierung wichtiger |

**Red Flags für ColBERT:**
- Budget-Cloud (Speicher teuer)
- Mobile/Edge Deployment
- >10M Dokumente (Speicher explodiert)

**Merksatz:**
"ColBERT wenn Qualität kritisch, Single-Vector wenn Skalierung kritisch!"

</details>

## 📊 Vergleiche & Varianten

### Wann nutze ich was?

| Use Case | Empfehlung | Warum? | Trade-off |
|----------|------------|--------|-----------|
| **E-Commerce Suche** | BM25 + Dense Hybrid | Keywords + Semantik | Zwei Indizes pflegen |
| **RAG System (Prototyping)** | Dense (Sentence-BERT) | Schnell zu starten | Exakte Matches fehlen |
| **RAG System (Production)** | Hybrid + Cross-Encoder Re-Ranking | Beste Qualität | Komplexität |
| **Legal/Medical Retrieval** | ColBERT | Token-Präzision wichtig | Hoher Speicher |
| **Millionen Docs, Latenz <50ms** | BM25 + Dense (keine Re-Ranking) | Speed kritisch | Qualitätsverlust |

### Decision Tree

```
Brauchst du exakte Keyword-Matches (IDs, Modellnummern)?
├─ Ja, nur Keywords
│   └─ BM25 (Sparse)
│
└─ Nein, Semantik wichtig
    ├─ Corpus klein (<100k)?
    │   ├─ Ja → High-Quality nötig?
    │   │   ├─ Ja → ColBERT
    │   │   └─ Nein → Dense (Bi-Encoder)
    │   │
    │   └─ Nein (>1M Docs)
    │       ├─ Latenz <50ms?
    │       │   ├─ Ja → Dense only (cached)
    │       │   └─ Nein → Dense + Cross-Encoder
    │       │
    │       └─ Budget für Hybrid?
    │           ├─ Ja → BM25 + Dense + Cross-Encoder
    │           └─ Nein → Dense only
```

## 🛠️ Tools & Frameworks

### Dense Embeddings

```python
# Sentence-Transformers (Standard)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["text1", "text2"])
```

**Models:**
- **all-MiniLM-L6-v2**: 384-dim, schnell, Englisch
- **paraphrase-multilingual-MiniLM-L12-v2**: 384-dim, multilingual
- **multilingual-e5-large**: 1024-dim, State-of-the-art

### Sparse Embeddings

```python
# BM25
from rank_bm25 import BM25Okapi

corpus_tokenized = [doc.split() for doc in corpus]
bm25 = BM25Okapi(corpus_tokenized)
scores = bm25.get_scores(query.split())
```

```python
# SPLADE
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained('naver/splade-cocondenser-ensembledistil')
tokenizer = AutoTokenizer.from_pretrained('naver/splade-cocondenser-ensembledistil')
```

### Multi-Vector (ColBERT)

```python
# ColBERT
from colbert import Indexer, Searcher

indexer = Indexer(checkpoint='colbert-ir/colbertv2.0')
indexer.index(name='my_index', collection=corpus)

searcher = Searcher(index='my_index')
results = searcher.search(query, k=10)
```

### Cross-Encoders

```python
# Cross-Encoder
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc] for doc in candidates]
scores = cross_encoder.predict(pairs)
```

**Häufige Stolpersteine:**

1. **Problem:** Cross-Encoder zu langsam
   ```python
   # Ursache: Auf zu viele Docs angewendet
   # Lösung: Nur Top-K Re-Ranken (K=50-100)
   ```

2. **Problem:** BM25 findet nichts bei Typos
   ```python
   # Ursache: Exakte String-Matches
   # Lösung: Fuzzy Matching oder Hybrid mit Dense
   ```

3. **Problem:** Dense rankt irrelevante Docs hoch
   ```python
   # Ursache: Zu "fuzzy" Matches
   # Lösung: Hybrid mit BM25 oder Cross-Encoder Re-Ranking
   ```

## 🚀 Was du jetzt kannst

**Verständnis:**
- ✓ Du verstehst die 4 Haupt-Architekturen und ihre mathematischen Grundlagen
- ✓ Du erkennst Trade-offs zwischen Speed, Qualität, Speicher
- ✓ Du verstehst warum Hybrid-Ansätze in Production Standard sind

**Praktische Fähigkeiten:**
- ✓ Du implementierst alle 4 Architekturen praktisch
- ✓ Du baust Two-Stage Retrieval (Bi-Encoder → Cross-Encoder)
- ✓ Du kombinierst BM25 + Dense für Hybrid-Search

**Kritisches Denken:**
- ✓ Du wählst Architektur basierend auf Requirements (Latenz, Corpus-Größe, Qualität)
- ✓ Du erkennst wann Single-Vector reicht vs. wann ColBERT/Cross-Encoder nötig
- ✓ Du debuggst Retrieval-Probleme (zu fuzzy? zu strict?)

**Nächste Schritte:**
- [ ] Baue Hybrid-System mit allen 4 Architekturen
- [ ] Benchmarke auf deinem eigenen Corpus
- [ ] Tune Hybrid-Gewichte (alpha-Parameter)

## 🔗 Weiterführende Themen

**Nächster logischer Schritt:**
→ [03-model-selection.md](03-model-selection.md) - **Embedding Spaces & Model Selection** (Kritisch: Warum Models verschiedene "Sprachen" sprechen!)

**Von Theorie zu Production:**
→ [04-vector-databases.md](04-vector-databases.md) - Vector DBs, Quantization, Deployment Patterns
→ [../../04-advanced/02-retrieval-optimization.md](../../04-advanced/02-retrieval-optimization.md) - Chunking Strategies, Two-Stage Retrieval, Hybrid Search

**Praktische Anwendung:**
→ [../../06-applications/rag-systems.md](../../06-applications/rag-systems.md) - Vollständiges RAG-System mit allem aus diesem Kapitel

**Verwandte Konzepte:**
- [../training/contrastive-learning.md](../training/contrastive-learning.md) - Wie Dense Models mit Contrastive Learning trainiert werden
- [../evaluation/metrics.md](../evaluation/metrics.md) - Recall@k, MRR, nDCG für Retrieval-Qualität
