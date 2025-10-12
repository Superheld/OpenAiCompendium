# Multi-Vector Embeddings: Deep Dive

## Inhaltsverzeichnis

1. [Was sind Multi-Vector Embeddings?](#was-sind-multi-vector-embeddings)
2. [ColBERT - Contextualized Late Interaction](#colbert)
3. [Late Interaction Mechanism](#late-interaction-mechanism)
4. [Poly-Encoders](#poly-encoders)
5. [Vor- und Nachteile](#vor--und-nachteile)
6. [Code-Beispiele](#code-beispiele)

---

## Was sind Multi-Vector Embeddings?

### Das Problem mit Single-Vector

**Standard Dense Embeddings (Bi-Encoder):**

```python
# Ein Vektor pro Text
doc = "Der Laborkühlschrank hat 280 Liter Volumen und ist DIN-konform"
embedding = model.encode(doc)  # [768] - EIN Vektor

# Problem: Alle Information muss in EINEN Vektor passen!
```

**Information Bottleneck:**

Lange Dokumente → Viele Details → 1 Vektor → **Informationsverlust**

```
Original Doc (500 Wörter):
"... Kühlschrank ... DIN 13277 ... 280 Liter ... +2°C bis +8°C ...
 SmartMonitoring ... WiFi ... Alarmierung ... Edelstahl ..."

↓ Komprimierung in 768 Dimensionen ↓

[0.23, -0.45, 0.12, ..., -0.34]  # Alle Details verschwommen!
```

### Multi-Vector Lösung

**Idee:** Nicht EIN Vektor, sondern VIELE Vektoren pro Dokument

```python
# Jedes Token bekommt eigenen Vektor
doc = "Laborkühlschrank 280 Liter DIN"

embeddings = [
    [0.12, 0.34, ...],  # "Laborkühlschrank" - 128-dim
    [0.45, -0.23, ...], # "280" - 128-dim
    [0.67, 0.89, ...],  # "Liter" - 128-dim
    [-0.34, 0.56, ...], # "DIN" - 128-dim
]  # 4 × 128 = 512 Werte total
```

**Vorteile:**
- ✅ Keine Kompression → kein Informationsverlust
- ✅ Token-Level Matching → präziser
- ✅ Kann längere Texte besser verarbeiten

**Nachteil:**
- ⚠️ Mehr Speicher (N Tokens × embedding_dim)
- ⚠️ Langsamere Suche (mehr Vergleiche)

### Single-Vector vs. Multi-Vector

**Visualisierung:**

```
Single-Vector (Bi-Encoder):
Doc: "Kühlschrank 280 Liter DIN"
     ↓
  [●] - EIN Punkt im Raum

Multi-Vector (z.B. ColBERT):
Doc: "Kühlschrank 280 Liter DIN"
     ↓
  [●] [●] [●] [●] - VIER Punkte im Raum
```

**Query Matching:**

```
Single-Vector:
Query Vector [●]  vs  Doc Vector [●]
     ↓
Cosine Similarity = 0.75

Multi-Vector:
Query: "280 Liter"
  [●] [●]

Doc: "Kühlschrank 280 Liter DIN"
  [●] [●] [●] [●]

Match:
  [●]────────[●]  "280" matches "280"
     [●]────────[●]  "Liter" matches "Liter"
```

**Multi-Vector findet exakte Token-Matches innerhalb semantischer Nähe!**

---

## ColBERT

### Contextualized Late Interaction over BERT

**Paper:** Khattab & Zaharia, 2020

**"Late Interaction"** - Was bedeutet das?

### Early vs. Late Interaction

**Early Interaction (Cross-Encoder):**

```
Query + Doc zusammen durch BERT
     ↓
[CLS] Query [SEP] Document [SEP]
     ↓
  [BERT]
     ↓
Relevance Score

Problem: Für jedes Doc-Query-Paar neu berechnen → SEHR langsam!
```

**No Interaction (Bi-Encoder / Standard Dense):**

```
Query         Document
  ↓              ↓
[BERT]        [BERT]
  ↓              ↓
[Vec]   vs   [Vec]
  ↓
Cosine Similarity

Vorteil: Docs vorher berechnen → schnell
Nachteil: Keine Query-Doc Interaktion → weniger genau
```

**Late Interaction (ColBERT):**

```
Query              Document
  ↓                   ↓
[BERT]             [BERT]
  ↓                   ↓
[Vec Vec Vec]   [Vec Vec Vec Vec]
       ↓                ↓
    Token-Level Matching ← "Late"
       ↓
  Relevance Score

Vorteil: Docs vorher berechnen + Token-Interaktion → Schnell UND genau!
```

### ColBERT Architektur

```
Query: "280 Liter Kühlschrank"
   ↓ Tokenize
[CLS] 280 Liter Kühlschrank [SEP] [Q] [Q] [Q] ...
                                     ↑
                                Query Padding (fixed length)
   ↓ BERT Encoder
[[vec], [vec], [vec], [vec], [vec], [vec], [vec], ...]
   ↓ Filter (nur echte Tokens, keine Padding)
[[vec1], [vec2], [vec3], [vec4]]
   ↓ L2 Normalize
Query Embeddings: 4 × 128-dim


Document: "Der Kirsch LABO-288 ist ein Laborkühlschrank mit 280 Liter"
   ↓ Tokenize
[CLS] Der Kirsch LABO ##- 288 ist ein Labor ##kühl ##schrank mit 280 Liter [SEP] [D] [D] ...
                                                                                   ↑
                                                                            Doc Padding
   ↓ BERT Encoder
   ↓ Filter + Normalize
Document Embeddings: 15 × 128-dim
```

**Key Points:**
- **128 Dimensionen** (statt 768) → kompakter
- **Fixed Length Padding** → gleiche Tensor-Größe für Batching
- **Separate Tokens:** `[Q]` für Query, `[D]` für Document
- **L2 Normalization** → Dot Product = Cosine

### Late Interaction: MaxSim

**Wie berechnet man Similarity zwischen zwei Token-Sets?**

**MaxSim (Maximum Similarity):**

Für jedes Query-Token: Finde das ähnlichste Doc-Token

```python
Query Tokens:    q1, q2, q3
Doc Tokens:      d1, d2, d3, d4, d5

Für q1: max(sim(q1, d1), sim(q1, d2), ..., sim(q1, d5))
Für q2: max(sim(q2, d1), sim(q2, d2), ..., sim(q2, d5))
Für q3: max(sim(q3, d1), sim(q3, d2), ..., sim(q3, d5))

Score = sum(all max similarities)
```

**Beispiel:**

```
Query: "280 Liter"
  q1 = vec("280")
  q2 = vec("Liter")

Doc: "Kühlschrank mit 280 Liter Volumen"
  d1 = vec("Kühlschrank")
  d2 = vec("mit")
  d3 = vec("280")
  d4 = vec("Liter")
  d5 = vec("Volumen")

MaxSim Berechnung:
  For q1 ("280"):
    sim(q1, d1) = 0.1  (Kühlschrank)
    sim(q1, d2) = 0.05 (mit)
    sim(q1, d3) = 0.98 (280) ← MAX!
    sim(q1, d4) = 0.2  (Liter)
    sim(q1, d5) = 0.15 (Volumen)
    → max = 0.98

  For q2 ("Liter"):
    sim(q2, d1) = 0.12
    sim(q2, d2) = 0.03
    sim(q2, d3) = 0.15
    sim(q2, d4) = 0.97 (Liter) ← MAX!
    sim(q2, d5) = 0.65 (Volumen - semantisch ähnlich)
    → max = 0.97

Score = 0.98 + 0.97 = 1.95
```

**Code:**

```python
import torch

def maxsim(query_embeddings, doc_embeddings):
    """
    query_embeddings: [num_query_tokens, dim]
    doc_embeddings: [num_doc_tokens, dim]
    """
    # Dot products (normalized → cosine)
    scores = torch.matmul(query_embeddings, doc_embeddings.T)
    # [num_query_tokens, num_doc_tokens]

    # Max über Doc-Dimension (für jedes Query-Token)
    max_scores = scores.max(dim=1).values  # [num_query_tokens]

    # Summe aller Max-Scores
    return max_scores.sum().item()
```

**Warum MaxSim und nicht Average?**

```
Query: "Laborkühlschrank DIN"

Doc A: "Laborkühlschrank nach DIN 13277"
  MaxSim: Beide Query-Tokens matchen perfekt → HIGH

Doc B: "Ein Laborkühlschrank ist ein Kühlgerät für Labore mit vielen Features und DIN Konformität"
  Average: Viele irrelevante Tokens verdünnen Signal → LOWER
  MaxSim: Findet trotzdem die richtigen Matches → HIGH

MaxSim ist robuster gegen lange Dokumente!
```

### ColBERT Training

**Dataset:** Query-Document-Paare mit Labels

```python
positive_pairs = [
    ("Laborkühlschrank 280L", "Der LABO-288 hat 280 Liter"),
]

negative_pairs = [
    ("Laborkühlschrank 280L", "Pommes Frites Rezept"),
]
```

**Loss:** Contrastive Loss mit In-Batch Negatives

```python
# Für jede Query im Batch
for query in batch:
    # Positive Doc
    pos_score = maxsim(query_emb, positive_doc_emb)

    # Negative Docs (alle anderen im Batch)
    neg_scores = [maxsim(query_emb, neg_doc_emb) for neg_doc in negatives]

    # Loss: Positive sollte höher sein als alle Negatives
    loss = -log(exp(pos_score) / (exp(pos_score) + sum(exp(neg_scores))))
```

**Training-Details:**
- **Knowledge Distillation:** Oft von Cross-Encoder lernen
- **Hard Negative Mining:** Schwierige Negatives samplen
- **Model:** BERT-base als Basis, dann finetuning

### ColBERT Indexing

**Problem:** Für jedes Doc N Token-Vektoren speichern!

```python
1M Docs × 200 Tokens/Doc × 128 dims × 4 bytes
= 102 GB!  ← Viel Speicher!
```

**Lösungen:**

**1. Kompression:**
```python
# Float32 → Float16
1M × 200 × 128 × 2 bytes = 51 GB

# Quantization (int8)
1M × 200 × 128 × 1 byte = 25 GB
```

**2. Clustering (Vector Quantization):**

```python
# Statt jeden Token-Vektor zu speichern:
# Clustere ähnliche Vektoren → Centroid-ID speichern

Original: [0.23, -0.45, ..., 0.12] (128 floats = 512 bytes)
Compressed: 42 (centroid ID = 4 bytes)

→ 128x Kompression!
```

**3. Pruning:**

```python
# Entferne unwichtige Tokens (Stopwords)
"Der Kühlschrank ist sehr kalt"
→ ["Kühlschrank", "kalt"]  # Nur 2 statt 5 Vektoren
```

### ColBERT Retrieval Pipeline

**2-Stage:**

**Stage 1: Candidate Generation** (schnell)
```python
# BM25 oder Standard Dense Retrieval
candidates = bm25.search(query, top_k=1000)
```

**Stage 2: Re-Ranking mit ColBERT** (langsam, aber genau)
```python
# Nur für Top-1000 Kandidaten
for doc in candidates:
    score = maxsim(query_embeddings, doc_embeddings[doc.id])

top_10 = sorted(candidates, key=lambda x: x.score, reverse=True)[:10]
```

**Warum 2-Stage?**
- ColBERT ist zu langsam für Millionen Docs (N Token-Vergleiche!)
- Aber sehr gut für Re-Ranking (präzise)

---

## Late Interaction Mechanism

### Warum "Late"?

**Timeline der Interaktion:**

```
Cross-Encoder (Earliest):
  Query + Doc → BERT → Score
  ↑
  Interaktion VOR Encoding (im BERT)

Bi-Encoder (No Interaction):
  Query → BERT → Vec
  Doc → BERT → Vec
  Vec ⟷ Vec → Score
  ↑
  Keine Interaktion (nur Cosine)

ColBERT (Late):
  Query → BERT → [Vec, Vec, Vec]
  Doc → BERT → [Vec, Vec, Vec, ...]
  [Vecs] ⟷ [Vecs] → MaxSim → Score
         ↑
  Interaktion NACH Encoding (in MaxSim)
```

**"Late" = Nach dem Encoding, aber vor dem finalen Score**

### Late Interaction Varianten

**1. MaxSim (ColBERT Standard):**
```python
score = sum(max(q_token ⋅ d_token for d_token in doc) for q_token in query)
```

**2. AvgMax (Average der Max-Scores):**
```python
score = mean(max(q_token ⋅ d_token for d_token in doc) for q_token in query)
```

**3. SumMax (nur Summe):**
```python
score = sum(max(q_token ⋅ d_token for d_token in doc) for q_token in query)
```

**4. Poly-Attention (gewichtet):**
```python
# Lerne Attention-Gewichte
weights = attention_module(query, doc)
score = sum(weights[i] * max(q[i] ⋅ d_token for d_token in doc) for i, q in enumerate(query))
```

### Visualisierung: Token Matching

```
Query: "280 Liter Kühlschrank"

Document: "Der LABO-288 Laborkühlschrank hat 280 Liter Volumen"

Matching Matrix (Similarity):

              Der  LABO  Labor  hat  280   Liter  Volumen
280          0.1  0.2   0.15   0.05 [0.98] 0.2   0.15
Liter        0.1  0.15  0.2    0.1  0.15  [0.97] 0.65
Kühlschrank  0.2  0.3   [0.85] 0.1  0.15   0.2   0.25

MaxSim:
  "280" → max = 0.98 (matched "280")
  "Liter" → max = 0.97 (matched "Liter")
  "Kühlschrank" → max = 0.85 (matched "Labor..." - prefix match!)

Total Score: 0.98 + 0.97 + 0.85 = 2.80
```

**Wichtig:** Jedes Query-Token findet sein bestes Match im Doc!

---

## Poly-Encoders

### Alternative zu ColBERT

**Paper:** Humeau et al., 2019

**Idee:** Statt N Vektoren pro Doc → M "Code"-Vektoren (M << N)

### Architektur

```
Document: "Laborkühlschrank mit 280 Liter nach DIN 13277" (8 Tokens)
   ↓
[BERT Encoding]
   ↓
Token Embeddings: 8 × 768
   ↓
[Attention Mechanism mit M=4 Codes]
   ↓
4 Code Vectors (statt 8 Token Vectors)
   ↓
[0.23, ...], [-0.45, ...], [0.12, ...], [0.89, ...]
```

**Vorteile gegenüber ColBERT:**
- Weniger Vektoren (4-64 statt N Tokens)
- Schneller bei Inference
- Weniger Speicher

**Nachteil:**
- Weniger präzise (Information-Kompression)

### Poly-Encoder Codes

**M Codes = M gelernte Attention-Queries**

```python
# M=4 Codes (gelernte Parameter)
code_vectors = [
    learnable_param_1,  # Fokus auf "Was?"
    learnable_param_2,  # Fokus auf "Wie viel?"
    learnable_param_3,  # Fokus auf "Standards?"
    learnable_param_4,  # Fokus auf "Technische Details"
]

# Attention über Token-Embeddings
for code in code_vectors:
    attention_weights = softmax(code @ token_embeddings.T)
    code_embedding = sum(attention_weights * token_embeddings)
```

**Jeder Code "extrahiert" einen Aspekt des Dokuments!**

### Matching

**Ähnlich wie ColBERT, aber mit Codes statt Tokens:**

```python
Query Embedding: [vec]  # Single Vector (wie Bi-Encoder)
Doc Code Embeddings: [vec1, vec2, vec3, vec4]  # M Vektoren

# Attention-basiertes Matching
attention = softmax(query_vec @ code_embeddings.T)
score = sum(attention * (query_vec @ code_embeddings.T))
```

### Poly-Encoder Varianten

**Poly-Encoder(m=1):** Bi-Encoder (kein Multi-Vector)
**Poly-Encoder(m=4-16):** Standard
**Poly-Encoder(m=N):** Fast wie ColBERT (alle Tokens)

**Trade-off:** Geschwindigkeit vs. Genauigkeit

---

## Vor- und Nachteile

### Vorteile von Multi-Vector Embeddings

✅ **Präzisere Matches**
```python
Query: "280 Liter Kühlschrank"
Doc: "... LABO-288 ... 280 Liter ..."

Single-Vector: Alles vermischt → fuzzy match
Multi-Vector: Findet exakt "280" und "Liter" Tokens → präzise!
```

✅ **Besseres Ranking**
```python
# BEIR Benchmark (Information Retrieval)
Bi-Encoder (Dense): nDCG@10 = 0.52
ColBERT: nDCG@10 = 0.58  ← +6% Verbesserung!
```

✅ **Längere Dokumente**
```python
# Single-Vector: 512 Tokens max (BERT Limit)
# ColBERT: Kann länger, weil pro Token embedded (weniger Kompression)
```

✅ **Interpretierbar**
```python
# Kann sehen WELCHE Tokens matchen
# "280" matched "280" mit Score 0.98
# "Liter" matched "Liter" mit Score 0.97
```

✅ **Robuster gegen Noise**
```python
Doc: "Kühlschrank ... <viel irrelevanter Text> ... 280 Liter"

Single-Vector: Relevante Info wird verdünnt
Multi-Vector: Findet trotzdem die relevanten Tokens!
```

---

### Nachteile von Multi-Vector Embeddings

❌ **Speicher-intensiv**
```python
Bi-Encoder: 1M Docs × 768 dims × 4 bytes = 3 GB
ColBERT: 1M Docs × 200 Tokens × 128 dims × 4 bytes = 102 GB

→ 34x mehr Speicher!
```

❌ **Langsamer**
```python
Bi-Encoder: 1 Cosine-Berechnung pro Doc
ColBERT: N × M Dot-Products + MaxSim

Für 1M Docs:
  Bi-Encoder: ~100ms
  ColBERT: ~5 seconds (mit Index) bis Minuten (ohne)
```

❌ **Komplexere Indexing**
```python
# Bi-Encoder: Standard FAISS/Annoy funktioniert
# ColBERT: Braucht spezielle Index-Strukturen
```

❌ **Training aufwändiger**
```python
# Mehr Parameter
# Mehr GPU-Speicher
# Längere Training-Zeit
```

❌ **Noch nicht weit verbreitet**
```python
# Weniger Libraries/Tools
# Weniger vortrainierte Models
# Community kleiner
```

---

### Wann Multi-Vector nutzen?

✅ **Perfekt für:**
- **Hohe Präzision wichtig** (Legal, Medical Search)
- **Längere Dokumente** (>512 Tokens)
- **Re-Ranking Stage** (Top-100 Kandidaten)
- **Erklärbarkeit** (welche Tokens matchen?)
- **Domain mit vielen Fachtermini** (exakte Matches wichtig)

❌ **Nicht ideal für:**
- **Latenz kritisch** (<50ms)
- **Millionen Dokumente** (ohne gutes Indexing)
- **Speicher-Constraints** (Edge Devices)
- **Simple Similarity Tasks** (Bi-Encoder reicht)

**Best Practice:**
```
Stage 1: Bi-Encoder oder BM25 (schnell, Top-1000)
   ↓
Stage 2: ColBERT Re-Ranking (präzise, Top-10)
```

---

## Code-Beispiele

### 1. ColBERT mit RAGatouille

```python
# RAGatouille ist einfachste ColBERT Library
from ragatouille import RAGPretrainedModel

# Model laden
model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Dokumente indizieren
documents = [
    "Laborkühlschrank LABO-288 mit 280 Liter Volumen nach DIN 13277",
    "Medikamentenkühlschrank HMF-4001 mit SmartMonitoring",
    "Gefrierschrank für Labor mit -40°C Temperaturbereich",
    "Pommes Frites Rezept"
]

# Index erstellen (speichert Token-Embeddings)
index_name = "my_products"
model.index(
    collection=documents,
    index_name=index_name,
    max_document_length=512,
    split_documents=False
)

# Suche
query = "Kühlschrank 280 Liter DIN"
results = model.search(
    query=query,
    index_name=index_name,
    k=3  # Top-3
)

print("ColBERT Results:")
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['content']}")
    print()
```

---

### 2. ColBERT von Grund auf (Inference)

```python
import torch
from transformers import BertModel, BertTokenizer

class SimpleColBERT:
    def __init__(self, model_name='bert-base-uncased', dim=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dim = dim

        # Linear Layer: 768 → 128
        self.linear = torch.nn.Linear(768, dim)

    def encode(self, texts, is_query=False):
        """Encode texts to multi-vector embeddings"""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # BERT Forward
        with torch.no_grad():
            outputs = self.bert(**encoded)
            embeddings = outputs.last_hidden_state  # [batch, seq_len, 768]

        # Linear Projection: 768 → 128
        embeddings = self.linear(embeddings)  # [batch, seq_len, 128]

        # L2 Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)

        # Mask out padding
        mask = encoded['attention_mask'].unsqueeze(-1)  # [batch, seq_len, 1]
        embeddings = embeddings * mask

        return embeddings

    def maxsim(self, query_emb, doc_emb):
        """
        query_emb: [1, query_len, dim]
        doc_emb: [1, doc_len, dim]
        """
        # Dot products
        scores = torch.matmul(query_emb, doc_emb.transpose(1, 2))
        # [1, query_len, doc_len]

        # Max over doc dimension
        max_scores = scores.max(dim=2).values  # [1, query_len]

        # Sum over query dimension
        return max_scores.sum().item()


# Usage
colbert = SimpleColBERT()

# Encode Query
query = "Kühlschrank 280 Liter"
query_emb = colbert.encode([query], is_query=True)

# Encode Documents
docs = [
    "Laborkühlschrank mit 280 Liter Volumen",
    "Pommes Frites sind lecker"
]
doc_embs = colbert.encode(docs)

# Score each document
for i, doc in enumerate(docs):
    score = colbert.maxsim(query_emb, doc_embs[i:i+1])
    print(f"Doc {i}: {score:.4f} - {doc}")
```

---

### 3. Token-Level Matching Visualisierung

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_token_matching(query_emb, doc_emb, query_tokens, doc_tokens):
    """Visualize which tokens match"""
    # Compute similarity matrix
    sim_matrix = torch.matmul(query_emb[0], doc_emb[0].T)  # [query_len, doc_len]

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        sim_matrix.cpu().numpy(),
        xticklabels=doc_tokens,
        yticklabels=query_tokens,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd'
    )
    plt.title('Token-Level Similarity (ColBERT)')
    plt.xlabel('Document Tokens')
    plt.ylabel('Query Tokens')
    plt.tight_layout()
    plt.show()

    # Print max matches
    print("\nMax Matches:")
    for i, q_token in enumerate(query_tokens):
        max_idx = sim_matrix[i].argmax().item()
        max_score = sim_matrix[i].max().item()
        print(f"'{q_token}' → '{doc_tokens[max_idx]}' (score: {max_score:.3f})")


# Example
colbert = SimpleColBERT()

query = "280 Liter Kühlschrank"
doc = "Laborkühlschrank mit 280 Liter Volumen"

query_emb = colbert.encode([query])
doc_emb = colbert.encode([doc])

query_tokens = colbert.tokenizer.tokenize(query)
doc_tokens = colbert.tokenizer.tokenize(doc)

visualize_token_matching(query_emb, doc_emb, query_tokens, doc_tokens)
```

---

### 4. Poly-Encoder Simplified

```python
import torch
import torch.nn as nn

class SimplePolyEncoder:
    def __init__(self, bert_model, num_codes=4, dim=768):
        self.bert = bert_model
        self.num_codes = num_codes

        # Learnable code vectors
        self.codes = nn.Parameter(torch.randn(num_codes, dim))

    def encode_document(self, doc):
        """Encode document to M code vectors"""
        # BERT Encoding
        outputs = self.bert(**doc)
        token_embs = outputs.last_hidden_state  # [batch, seq_len, 768]

        # Attention with codes
        code_embeddings = []
        for code in self.codes:
            # Attention weights
            attention = torch.softmax(
                torch.matmul(code, token_embs.transpose(1, 2)),  # [batch, seq_len]
                dim=-1
            )

            # Weighted sum
            code_emb = torch.matmul(
                attention.unsqueeze(1),  # [batch, 1, seq_len]
                token_embs               # [batch, seq_len, 768]
            ).squeeze(1)  # [batch, 768]

            code_embeddings.append(code_emb)

        return torch.stack(code_embeddings, dim=1)  # [batch, num_codes, 768]

    def encode_query(self, query):
        """Encode query to single vector (like Bi-Encoder)"""
        outputs = self.bert(**query)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token

    def score(self, query_emb, doc_code_embs):
        """Score query vs document codes"""
        # Dot products
        scores = torch.matmul(query_emb, doc_code_embs.transpose(1, 2))
        # [batch, 1, num_codes]

        # Attention over codes
        attention = torch.softmax(scores, dim=-1)

        # Weighted sum
        final_score = (attention * scores).sum()

        return final_score.item()
```

---

### 5. Two-Stage Retrieval mit ColBERT

```python
from rank_bm25 import BM25Okapi
from ragatouille import RAGPretrainedModel

class HybridRetriever:
    def __init__(self, documents):
        self.documents = documents

        # Stage 1: BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Stage 2: ColBERT
        self.colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        self.colbert.index(
            collection=documents,
            index_name="hybrid_index",
            max_document_length=512
        )

    def search(self, query, k=10):
        """Two-stage retrieval"""
        # Stage 1: BM25 (fast, broad)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Top-100 candidates
        top_100_indices = bm25_scores.argsort()[-100:][::-1]
        candidates = [self.documents[i] for i in top_100_indices]

        # Stage 2: ColBERT Re-Ranking (slow, precise)
        # Note: RAGatouille searches entire index, so we'd need custom implementation
        # For demo, just use ColBERT directly
        results = self.colbert.search(
            query=query,
            index_name="hybrid_index",
            k=k
        )

        return results


# Usage
docs = [
    "Laborkühlschrank LABO-288 mit 280 Liter",
    "Medikamentenkühlschrank nach DIN 13277",
    # ... more docs
]

retriever = HybridRetriever(docs)
results = retriever.search("Kühlschrank 280 Liter DIN", k=5)

for r in results:
    print(f"{r['score']:.4f} - {r['content']}")
```

---

## Zusammenfassung

### Key Takeaways

1. **Multi-Vector = N Vektoren pro Dokument** (statt 1)
2. **ColBERT:** Token-Level Embeddings + Late Interaction (MaxSim)
3. **Late Interaction:** Matching NACH Encoding (Hybrid aus Bi-Encoder Speed + Cross-Encoder Precision)
4. **MaxSim:** Jedes Query-Token findet bestes Match im Doc
5. **Poly-Encoder:** Kompromiss (M Codes statt N Tokens)
6. **Vorteile:** Präziser, längere Docs, interpretierbar
7. **Nachteile:** Mehr Speicher, langsamer, komplexer
8. **Best Practice:** Two-Stage (BM25/Dense → ColBERT Re-Ranking)

### Vergleich

| Methode | Vektoren/Doc | Latenz | Genauigkeit | Speicher |
|---------|--------------|--------|-------------|----------|
| **Bi-Encoder** | 1 | ⚡⚡⚡ | ⭐⭐ | ✅✅✅ |
| **Poly-Encoder** | 4-16 | ⚡⚡ | ⭐⭐⭐ | ✅✅ |
| **ColBERT** | N (50-200) | ⚡ | ⭐⭐⭐⭐ | ✅ |
| **Cross-Encoder** | - | 💤 | ⭐⭐⭐⭐⭐ | ✅✅ |

---

## Nächste Schritte

- [05-CROSS-ENCODERS.md](05-CROSS-ENCODERS.md) - Noch präziser, aber langsamer
- [06-HYBRID-APPROACHES.md](06-HYBRID-APPROACHES.md) - Kombiniere Dense + Sparse + Multi-Vector
- [08-VECTOR-DATABASES.md](08-VECTOR-DATABASES.md) - Indexing für Multi-Vector

---

## Weiterführende Ressourcen

**Papers:**
- Khattab & Zaharia 2020: "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"
- Humeau et al. 2019: "Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring"

**Code:**
- [ColBERT Official](https://github.com/stanford-futuredata/ColBERT)
- [RAGatouille](https://github.com/bclavie/RAGatouille) - Einfachste ColBERT Library

**Benchmarks:**
- [BEIR Leaderboard](https://github.com/beir-cellar/beir) - ColBERT Performance
