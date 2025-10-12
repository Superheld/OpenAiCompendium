# Dense Embeddings: Deep Dive

## 🎯 Ziel
Beherrsche Dense Embeddings von Transformer-Architektur bis zu Sentence-Transformers - der Industriestandard für semantische Suche und RAG-Systeme.

## 📖 Geschichte & Kontext

Dense Embeddings revolutionierte NLP mit der Transformer-Architektur. Von BERTs bidirektionaler Innovation bis zu modernen Sentence-Transformers - sie sind das Herzstück aller modernen AI-Anwendungen.

**Meilensteine:**
- **2017**: Transformer-Architektur ("Attention is All You Need")
- **2018**: BERT - bidirektionale kontextualisierte Embeddings
- **2019**: Sentence-BERT - optimiert für Sentence-Level Similarity
- **2024**: E5, BGE - speziell für Retrieval optimierte Dense Models

## 🧮 Konzept & Theorie

## Was sind Dense Embeddings?

### Definition

**Dense Embeddings** sind kontinuierliche Vektorrepräsentationen, bei denen **alle Dimensionen** mit Werten gefüllt sind (keine Nullen).

```python
# Dense Embedding (384 Dimensionen)
[0.234, -0.456, 0.123, 0.789, -0.234, 0.567, ..., -0.123]
# Alle 384 Werte sind != 0 (meist zwischen -1 und 1)
```

**Im Gegensatz zu Sparse:**
```python
# Sparse Embedding (30.000 Dimensionen)
[0, 0, 0, 3.2, 0, 0, 0, 1.5, 0, 0, 0, ..., 0]
# Meiste Werte = 0, nur wenige != 0
```

### Eigenschaften

- **Größe:** Typisch 384, 768, 1024 Dimensionen
- **Wertebereich:** Meist -1 bis +1 (normalisiert)
- **Speicher:** ~3KB pro Embedding (768 dims × 4 bytes float32)
- **Semantik:** Encodiert Bedeutung, nicht Keywords

### Wie entstehen Dense Embeddings?

**Von Text zu Vektor - die Pipeline:**

```
1. Tokenisierung
"Laborkühlschrank" → ["Labor", "##kühl", "##schrank"]

2. Token-zu-ID
["Labor", "##kühl", "##schrank"] → [2847, 19384, 8473]

3. Embedding Lookup
[2847, 19384, 8473] → [[0.12, -0.34, ...], [0.56, 0.23, ...], [-0.78, 0.45, ...]]

4. Transformer Layers
Contextualisierung durch Self-Attention

5. Pooling
3 Token-Embeddings → 1 Sentence-Embedding

6. Normalisierung
Final: [0.234, -0.456, ..., -0.123]
```

**Schauen wir uns jeden Schritt genau an:**

---

## Transformer-Architektur

### Das Herzstück moderner Embeddings

**Paper:** "Attention is All You Need" (Vaswani et al., 2017)

### Self-Attention Mechanism

**Problem:** Wie versteht ein Model Kontext?

**Beispiel:**
```
"Die Bank am Fluss ist schön."     # Bank = Sitzbank
"Die Bank hat heute geschlossen."  # Bank = Geldinstitut
```

Das Wort "Bank" braucht unterschiedliche Embeddings je nach Kontext!

**Lösung: Self-Attention**

Jedes Wort "schaut" auf alle anderen Wörter und entscheidet, welche wichtig sind.

```
Satz: "Laborkühlschrank für Medikamente"

Token: "Medikamente"
Attention-Gewichte auf andere Wörter:
- "Labor"        : 0.1  (etwas relevant)
- "kühlschrank"  : 0.7  (sehr relevant!)
- "für"          : 0.05 (wenig relevant)
- "Medikamente"  : 0.15 (selbst)
```

### Mathematik (vereinfacht)

**Self-Attention in 3 Schritten:**

**1. Query, Key, Value Matrizen erstellen**
```python
Q = input × W_query   # "Was suche ich?"
K = input × W_key     # "Was biete ich?"
V = input × W_value   # "Was ist mein Inhalt?"
```

**2. Attention Scores berechnen**
```python
scores = Q @ K.T / sqrt(d_k)  # Wie ähnlich sind Query und Keys?
attention_weights = softmax(scores)  # Normalisiere zu Wahrscheinlichkeiten
```

**3. Gewichtete Summe der Values**
```python
output = attention_weights @ V  # Mische Values basierend auf Relevanz
```

**Visualisierung:**

```
Input: "Der Kühlschrank ist kalt"

        Der    Kühl-   schrank   ist    kalt
Der    [0.2    0.1     0.1      0.3    0.3]   ← Attention-Gewichte
Kühl-  [0.1    0.3     0.4      0.1    0.1]
schrank[0.1    0.4     0.3      0.1    0.1]   ← "schrank" achtet auf "Kühl-"!
ist    [0.2    0.1     0.1      0.4    0.2]
kalt   [0.2    0.15    0.15     0.2    0.3]
```

**"schrank" hat hohe Attention auf "Kühl-"** → Kontext wird verstanden!

### Multi-Head Attention

**Problem:** Ein Attention-Mechanismus ist limitiert

**Lösung:** Mehrere parallel (z.B. 12 Heads)

- Head 1: Achtet auf Syntax (Subjekt-Verb-Beziehungen)
- Head 2: Achtet auf Semantik (ähnliche Bedeutung)
- Head 3: Achtet auf Named Entities
- ...
- Head 12: Achtet auf Negationen

**Jeder Head lernt andere Muster!**

```python
# Pseudo-Code
outputs = []
for head in range(12):
    Q, K, V = create_qkv(input, head)
    attention = self_attention(Q, K, V)
    outputs.append(attention)

final_output = concatenate(outputs) @ W_output
```

### Feed-Forward Network

Nach Self-Attention kommt noch ein FFN:

```python
# Self-Attention Output
x_attention = multi_head_attention(x)

# Feed-Forward
x_ff = linear(relu(linear(x_attention)))

# Residual Connection + Layer Norm
output = layer_norm(x + x_ff)
```

**Warum?** FFN transformiert die Attention-Outputs nochmal non-linear.

### Transformer Block (komplett)

```
Input Embedding
    ↓
[Multi-Head Self-Attention]
    ↓
[Add & Normalize]  ← Residual Connection
    ↓
[Feed-Forward Network]
    ↓
[Add & Normalize]  ← Residual Connection
    ↓
Output
```

**Typisch:** 12-24 solcher Blöcke gestackt!

---

## BERT

### Bidirectional Encoder Representations from Transformers

**Paper:** Devlin et al., 2018

### Was macht BERT besonders?

**Bidirektional:** Schaut in beide Richtungen gleichzeitig

```
Andere Models (GPT):          BERT:
"Der Kühlschrank ___"         "Der ___ ist kalt"
     ←                              ↔
Nur links schauen             Links UND rechts!
```

**Vorteil:** Besseres Kontext-Verständnis

### BERT-Architektur

```
[CLS] Der Kühlschrank ist kalt [SEP]
  ↓     ↓       ↓       ↓   ↓    ↓
┌─────────────────────────────────┐
│      Token Embeddings            │
│   + Positional Encodings         │ ← Wo steht das Token?
│   + Segment Embeddings           │ ← Satz A oder B?
└─────────────────────────────────┘
            ↓
  ┌──────────────────┐
  │ Transformer × 12 │  ← BERT-Base
  └──────────────────┘
            ↓
   768-dim Embeddings für jedes Token
```

**Spezielle Tokens:**
- `[CLS]` - Classification Token (am Anfang)
- `[SEP]` - Separator (zwischen Sätzen)

### BERT Training (Pre-Training)

**Zwei Tasks gleichzeitig:**

**1. Masked Language Modeling (MLM)**

```
Original: "Der Laborkühlschrank ist sehr kalt"
Masked:   "Der [MASK] ist sehr kalt"

Task: Vorhersage des maskierten Wortes
Model Output: "Laborkühlschrank" ✓
```

- 15% der Tokens werden maskiert
- Model lernt Kontext-Verständnis

**2. Next Sentence Prediction (NSP)**

```
Sentence A: "Der Kühlschrank ist defekt."
Sentence B: "Wir müssen ihn reparieren."
Label: IsNext ✓

Sentence A: "Der Kühlschrank ist defekt."
Sentence B: "Bananen sind gelb."
Label: NotNext ✓
```

**Training-Daten:** Wikipedia, BookCorpus (3.3 Milliarden Tokens!)

### BERT-Varianten

**BERT-Base:**
- 12 Transformer-Layers
- 768 Dimensionen
- 110M Parameter
- Schneller

**BERT-Large:**
- 24 Transformer-Layers
- 1024 Dimensionen
- 340M Parameter
- Genauer, aber langsamer

**Andere Sprachen:**
- **mBERT** - Multilingual (104 Sprachen)
- **GBERT** - Deutsch-spezifisch
- **CamemBERT** - Französisch
- etc.

### Problem mit BERT für Sentence Embeddings

**BERT wurde für Token-Klassifikation trainiert, nicht Sentence-Similarity!**

```python
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Problem: Welchen Output nehmen?
inputs = tokenizer("Der Kühlschrank", return_tensors='pt')
outputs = model(**inputs)

# outputs.last_hidden_state: [1, seq_len, 768]
# → Ein Vektor PRO TOKEN, nicht pro Satz!
```

**Lösung:** Sentence-BERT (SBERT)

---

## Sentence-Transformers

### Von BERT zu SBERT

**Paper:** Reimers & Gurevych, 2019

**Problem:** BERT gibt Token-Level Embeddings, wir brauchen Sentence-Level!

**Naive Lösung: Mean Pooling**
```python
# Nimm alle Token-Embeddings und berechne Durchschnitt
sentence_embedding = mean(token_embeddings)
```

**Aber:** BERT wurde nicht darauf trainiert! → Schlechte Sentence-Similarity

**SBERT-Lösung:** BERT mit Siamese Networks finetunen

### Siamese Network Architecture

**Training für Sentence Similarity:**

```
Sentence A: "Laborkühlschrank 280L"
Sentence B: "Medikamentenkühlschrank 280 Liter"
Label: Ähnlich (0.9)

    ┌─────────────┐         ┌─────────────┐
    │   BERT      │         │   BERT      │  ← Gleiche Gewichte!
    │  (shared)   │         │  (shared)   │
    └─────────────┘         └─────────────┘
          ↓                       ↓
    [0.23, -0.45,...]       [0.25, -0.43,...]
          ↓                       ↓
       Cosine Similarity = 0.89
          ↓
    Loss: |0.89 - 0.9| = 0.01  ← Minimize!
```

**Zwei identische BERT-Modelle** (shared weights) → Embeddings → Loss berechnen

### Training-Objectives

**1. Regression (MSE Loss)**
```python
# Dataset: Sentence-Paare mit Similarity-Scores
("Kühlschrank", "Kühlaggregat") → 0.7
("Kühlschrank", "Banane")       → 0.1

loss = (predicted_similarity - true_similarity)²
```

**2. Contrastive Loss**
```python
# Positive Paare (ähnlich) sollen nah sein
# Negative Paare (unterschiedlich) sollen weit auseinander

if similar:
    loss = distance²
else:
    loss = max(0, margin - distance)²
```

**3. Triplet Loss**
```python
# Anchor, Positive, Negative
anchor = "Laborkühlschrank"
positive = "Medikamentenkühlschrank"  # Ähnlich
negative = "Pommes Frites"             # Unterschiedlich

loss = max(0,
    distance(anchor, positive) - distance(anchor, negative) + margin
)

# Ziel: anchor näher an positive als an negative (um mindestens margin)
```

**Visualisierung Triplet Loss:**
```
Vorher:
    Anchor •

    Positive •        Negative •

Nachher:
    Anchor •
    Positive •                       • Negative

→ Positive näher, Negative weiter weg
```

### Sentence-Transformers Library

**Installation:**
```bash
pip install sentence-transformers
```

**Verwendung:**
```python
from sentence_transformers import SentenceTransformer

# Model laden (viele vortrainierte verfügbar)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Embeddings generieren
sentences = [
    "Laborkühlschrank mit 280 Liter Volumen",
    "Medikamentenkühlschrank 280L",
    "Pommes Frites sind lecker"
]

embeddings = model.encode(sentences)
# Output: (3, 384) numpy array

# Similarity berechnen
from sentence_transformers import util

similarities = util.cos_sim(embeddings[0], embeddings[1:])
print(similarities)
# [[0.85, 0.12]]  ← Kühlschränke ähnlich, Pommes nicht
```

**Vorteile:**
- ✅ Direkt einsatzbereit
- ✅ Hunderte vortrainierte Models
- ✅ Optimiert für Sentence-Similarity
- ✅ Einfaches Fine-Tuning

---

## Pooling-Strategien

### Von Token-Embeddings zu Sentence-Embeddings

**Problem:** BERT gibt `[seq_len, 768]`, wir brauchen `[768]`

**Lösung:** Pooling - kombiniere Token-Embeddings zu einem

### 1. Mean Pooling (am häufigsten!)

**Idee:** Durchschnitt aller Token-Embeddings

```python
def mean_pooling(token_embeddings, attention_mask):
    """
    token_embeddings: [batch_size, seq_len, hidden_dim]
    attention_mask: [batch_size, seq_len]  # 1 für echte Tokens, 0 für Padding
    """
    # Erweitere attention_mask für Broadcasting
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

    # Summiere nur echte Tokens (ignoriere Padding)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # Durchschnitt
    return sum_embeddings / sum_mask
```

**Beispiel:**
```
Tokens: [CLS] Der Kühlschrank [SEP] [PAD] [PAD]
Embeddings:
  [0.1, 0.2, 0.3]  ← [CLS]
  [0.4, 0.5, 0.6]  ← Der
  [0.7, 0.8, 0.9]  ← Kühlschrank
  [0.2, 0.3, 0.4]  ← [SEP]
  [0.0, 0.0, 0.0]  ← [PAD] (ignoriert)
  [0.0, 0.0, 0.0]  ← [PAD] (ignoriert)

Mean Pooling:
  (0.1+0.4+0.7+0.2)/4, (0.2+0.5+0.8+0.3)/4, (0.3+0.6+0.9+0.4)/4
= [0.35, 0.45, 0.55]
```

**Vorteile:**
- ✅ Alle Tokens tragen bei
- ✅ Standard für Sentence-Transformers
- ✅ Funktioniert robust

**Nachteile:**
- ⚠️ Wichtige Wörter nicht stärker gewichtet

---

### 2. CLS Pooling

**Idee:** Nutze nur das `[CLS]`-Token

```python
def cls_pooling(token_embeddings):
    # Erstes Token ist immer [CLS]
    return token_embeddings[:, 0, :]
```

**BERT wurde so trainiert:** `[CLS]` soll ganze Sequence repräsentieren

**Vorteile:**
- ✅ Einfach
- ✅ Schnell

**Nachteile:**
- ⚠️ Nur gut wenn Model darauf trainiert (nicht immer der Fall!)
- ⚠️ Verschwendet Info aus anderen Tokens

**Wann nutzen?**
- Bei BERT-basierten Models die auf NSP trainiert wurden
- Für Klassifikation (nicht Similarity!)

---

### 3. Max Pooling

**Idee:** Nimm Maximum über alle Tokens (pro Dimension)

```python
def max_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

    # Setze Padding auf sehr kleinen Wert
    token_embeddings[input_mask_expanded == 0] = -1e9

    # Max über Sequence-Dimension
    return torch.max(token_embeddings, dim=1)[0]
```

**Beispiel:**
```
Token Embeddings:
  Dim 0: [0.1, 0.4, 0.7, 0.2]  → Max = 0.7
  Dim 1: [0.2, 0.5, 0.8, 0.3]  → Max = 0.8
  Dim 2: [0.3, 0.6, 0.9, 0.4]  → Max = 0.9

Max Pooling: [0.7, 0.8, 0.9]
```

**Vorteile:**
- ✅ Behält stärkste Signale

**Nachteile:**
- ⚠️ Kann zu "spiky" Embeddings führen
- ⚠️ Seltener verwendet als Mean

---

### 4. Weighted Mean (Attention-based)

**Idee:** Gewichte Tokens basierend auf Wichtigkeit

```python
def weighted_mean_pooling(token_embeddings, attention_weights):
    # attention_weights aus letzter Layer
    weighted = token_embeddings * attention_weights.unsqueeze(-1)
    return weighted.sum(dim=1)
```

**Vorteil:** Wichtige Wörter zählen mehr

**Nachteil:** Komplexer, nicht immer besser als Mean

---

### Vergleich der Pooling-Strategien

```python
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('bert-base-uncased')
text = "Der Laborkühlschrank ist sehr kalt"

# Token-Embeddings holen
inputs = model.tokenize([text])
with torch.no_grad():
    features = model.forward(inputs)
    token_embeddings = features['token_embeddings']  # [1, seq_len, 768]
    attention_mask = features['attention_mask']       # [1, seq_len]

# Mean Pooling
mean_emb = mean_pooling(token_embeddings, attention_mask)

# CLS Pooling
cls_emb = token_embeddings[:, 0, :]

# Max Pooling
max_emb = max_pooling(token_embeddings, attention_mask)

print("Mean:", mean_emb.shape)  # [1, 768]
print("CLS:", cls_emb.shape)    # [1, 768]
print("Max:", max_emb.shape)    # [1, 768]

# Sind sie ähnlich?
from sentence_transformers import util
print("Mean vs CLS:", util.cos_sim(mean_emb, cls_emb).item())
print("Mean vs Max:", util.cos_sim(mean_emb, max_emb).item())
```

**Empfehlung:** **Mean Pooling** (Standard in Sentence-Transformers)

---

## Training-Methoden

### Wie lernt ein Embedding-Model?

**Ziel:** Semantisch ähnliche Sätze sollen nahe Vektoren haben

### 1. Contrastive Learning

**Prinzip:** Positive Paare näher bringen, Negative weiter auseinander

**Dataset:**
```python
positive_pairs = [
    ("Kühlschrank", "Kühlaggregat"),
    ("Medikament", "Arznei"),
]

negative_pairs = [
    ("Kühlschrank", "Banane"),
    ("Medikament", "Auto"),
]
```

**Loss:**
```python
for (anchor, positive) in positive_pairs:
    emb_a = model.encode(anchor)
    emb_p = model.encode(positive)

    loss_positive = distance(emb_a, emb_p)  # Minimize

for (anchor, negative) in negative_pairs:
    emb_a = model.encode(anchor)
    emb_n = model.encode(negative)

    loss_negative = max(0, margin - distance(emb_a, emb_n))  # Push apart
```

---

### 2. Triplet Loss (härter)

**Prinzip:** Anchor näher an Positive als an Negative (um mindestens Margin)

```python
triplets = [
    ("Kühlschrank", "Kühlaggregat", "Banane"),
    #   anchor        positive        negative
]

for (anchor, positive, negative) in triplets:
    emb_a = model.encode(anchor)
    emb_p = model.encode(positive)
    emb_n = model.encode(negative)

    loss = max(0,
        distance(emb_a, emb_p) - distance(emb_a, emb_n) + margin
    )
```

**Visualisierung:**
```
Gutes Triplet:
  Anchor •---• Positive          • Negative
  (Margin erfüllt, kein Loss)

Schlechtes Triplet:
  Anchor •-------• Positive
                • Negative
  (Negative zu nah! Loss > 0)
```

**Hard Negative Mining:**

Nicht alle Negative sind gleich lehrreich:

```python
# Easy Negative (nutzlos)
("Kühlschrank", "Pommes")  # Offensichtlich unterschiedlich

# Hard Negative (sehr lehrreich!)
("Laborkühlschrank", "Gefrierschrank")  # Ähnlich, aber nicht gleich!
```

**Mining-Strategie:**
```python
# Finde härteste Negatives im Batch
for anchor in batch:
    # Alle anderen als Kandidaten
    candidates = [x for x in batch if x != anchor]

    # Sortiere nach Similarity
    similarities = [cos_sim(anchor, c) for c in candidates]

    # Härtestes = ähnlichstes (aber falsche Klasse)
    hardest_negative = candidates[argmax(similarities)]
```

---

### 3. Multiple Negatives Ranking Loss

**Prinzip:** In einem Batch, jedes Positive-Paar, alle anderen sind Negatives

```python
batch = [
    ("Query 1", "Positive Doc 1"),
    ("Query 2", "Positive Doc 2"),
    ("Query 3", "Positive Doc 3"),
]

# Für Query 1:
#   - Positive: Doc 1
#   - Negatives: Doc 2, Doc 3 (alle anderen Docs im Batch!)
```

**Loss:**
```python
# Für jeden Query
scores = cos_sim(query, all_docs_in_batch)  # [batch_size]
labels = index_of_positive_doc              # z.B. 0 (erstes Doc)

loss = cross_entropy(scores, labels)  # Maximiere Score für richtiges Doc
```

**Vorteil:** Sehr effizient! Batch als implizites Hard Negative Mining

**Verwendet von:** Sentence-Transformers, E5, BGE, ...

---

### 4. Knowledge Distillation

**Prinzip:** Kleines Model lernt von großem Model

```python
# Teacher (groß, genau, langsam)
teacher = SentenceTransformer('all-mpnet-base-v2')  # 768 dims, 420M params

# Student (klein, schneller, weniger genau)
student = SentenceTransformer('all-MiniLM-L6-v2')   # 384 dims, 22M params

# Training
for sentence in dataset:
    teacher_emb = teacher.encode(sentence)  # "Gold Standard"
    student_emb = student.encode(sentence)

    loss = mse(student_emb, teacher_emb)  # Student imitiert Teacher
```

**Vorteil:** Kleines Model fast so gut wie großes, aber viel schneller!

**Beispiel:** MiniLM (22M params) erreicht 95% Performance von RoBERTa-Large (355M params)

---

## Model-Familien

### Übersicht der wichtigsten Dense Embedding Models

### 1. BERT-Familie

**BERT-Base / BERT-Large**
- Basis für viele andere
- 768 / 1024 dims
- Englisch oder Multilingual

**Varianten:**
- **RoBERTa** - Robustly optimized BERT (besseres Training)
- **DeBERTa** - Decoding-enhanced BERT (bessere Attention)
- **ALBERT** - Parameter-Sharing (kleiner)

### 2. Sentence-Transformers Familie

**all-MiniLM-L6-v2** ⭐ Empfehlung für Anfang
- 384 Dimensionen
- 22M Parameter
- Sehr schnell
- Englisch
- [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**all-mpnet-base-v2**
- 768 Dimensionen
- 110M Parameter
- Beste Qualität (SBERT)
- Englisch

**paraphrase-multilingual-MiniLM-L12-v2** ⭐ Für Deutsch
- 384 Dimensionen
- Multilingual (50+ Sprachen)
- Gut für deutsche Texte

**paraphrase-multilingual-mpnet-base-v2**
- 768 Dimensionen
- Multilingual
- Höhere Qualität als MiniLM

### 3. E5-Familie (Microsoft)

**intfloat/multilingual-e5-large** ⭐⭐ Top-Wahl für Deutsch
- 1024 Dimensionen
- Multilingual
- State-of-the-art Performance
- Trainiert auf 1 Billion Token-Paare!

**Besonderheit:** Nutzt Prefix für Query vs. Document

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-large')

# WICHTIG: Prefix verwenden!
query_emb = model.encode("query: Laborkühlschrank 280L")
doc_emb = model.encode("passage: Der Kirsch LABO-288 ist ein Laborkühlschrank...")

similarity = util.cos_sim(query_emb, doc_emb)
```

**Warum Prefix?**
- Model wurde so trainiert
- Unterscheidet Query (kurz) von Document (lang)
- Bessere Performance!

**Varianten:**
- **e5-small** - 384 dims, schnell
- **e5-base** - 768 dims
- **e5-large** - 1024 dims, beste Qualität

### 4. BGE-Familie (BAAI - Beijing Academy of AI)

**BAAI/bge-large-en-v1.5**
- 1024 Dimensionen
- Englisch
- Sehr gut für RAG

**BAAI/bge-m3** ⭐ Multilingual + Multi-Granularity
- Multilingual (100+ Sprachen)
- Dense + Sparse + Multi-Vector in einem!
- Sehr vielseitig

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')

# Dense Embedding
dense_emb = model.encode("Text")

# Kann auch Sparse und Multi-Vector (advanced)
```

### 5. Deutsche Models

**T-Systems-onsite/german-roberta-sentence-transformer-v2** ⭐ Beste rein deutsche Option
- 768 Dimensionen
- Nur Deutsch
- Sehr gut bei Fachsprache

**deutsche-telekom/gbert-large-paraphrase-cosine**
- 1024 Dimensionen
- Nur Deutsch

### 6. Spezialisierte Models

**allenai/specter2** - Scientific Papers
**microsoft/codebert-base** - Code
**emilyalsentzer/Bio_ClinicalBERT** - Medical (Englisch)

---

## Vor- und Nachteile

### Vorteile von Dense Embeddings

✅ **Semantisches Verständnis**
```python
# Findet Synonyme, Paraphrasen
query = "Kühlaggregat"
# Findet: "Kühlschrank", "Kühlgerät", "Kältemaschine"
```

✅ **Cross-Lingual**
```python
# Multilingual Models
query_de = model.encode("Kühlschrank")
query_en = model.encode("refrigerator")
# Ähnliche Embeddings! Cross-Language Search möglich
```

✅ **Robustheit gegen Typos**
```python
query = "Kühlschronk"  # Typo
# Findet trotzdem "Kühlschrank" (ähnlicher Vektor)
```

✅ **Kontextuelles Verständnis**
```python
"Bank am Fluss" vs "Bank hat geschlossen"
# Unterschiedliche Embeddings trotz gleichem Wort!
```

---

### Nachteile von Dense Embeddings

❌ **Exakte Matches können fehlen**
```python
query = "LABO-288-PRO"  # Modellnummer
# Könnte schlechter matchen als Sparse (BM25)
```

❌ **Blackbox**
```python
# Warum rankt Doc A höher als Doc B?
# Schwer zu erklären (768 Dimensionen!)
```

❌ **Rechenintensiv**
```python
# Embedding-Generierung braucht GPU
# Für 1M Dokumente: ~10 Minuten auf GPU
```

❌ **Speicher**
```python
# 1M Dokumente × 768 dims × 4 bytes = ~3GB
# Sparse (BM25) braucht weniger
```

❌ **Domain-Shift**
```python
# Trainiert auf Wikipedia
# Könnte schlechter sein für medizinische Fachsprache
# → Lösung: Fine-Tuning (separates Kapitel)
```

---

### Wann Dense Embeddings?

✅ **Perfekt für:**
- Semantische Suche
- Frage-Antwort-Systeme
- Paraphrasen-Erkennung
- Cross-Lingual Retrieval
- Ähnliche Dokumente finden

❌ **Nicht ideal für:**
- Exakte Keyword-Suche (→ Sparse)
- Modellnummern, IDs, SKUs (→ Sparse oder Hybrid)
- Sehr lange Dokumente >512 Tokens (→ Chunking oder Sparse)
- Wenn Erklärbarkeit wichtig (→ BM25 zeigt Keyword-Matches)

**Best Practice:** Hybrid (Dense + Sparse) für Production!

---

## Code-Beispiele

### 1. Basic Usage

```python
from sentence_transformers import SentenceTransformer, util

# Model laden
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Texte
texts = [
    "Laborkühlschrank mit 280 Liter Volumen",
    "Medikamentenkühlschrank 280L DIN-konform",
    "Gefrierschrank -20°C",
    "Pommes Frites"
]

# Embeddings generieren
embeddings = model.encode(texts, convert_to_tensor=True)
print(embeddings.shape)  # (4, 384)

# Similarity Matrix
similarities = util.cos_sim(embeddings, embeddings)
print(similarities)
# [[1.00, 0.85, 0.62, 0.12],
#  [0.85, 1.00, 0.58, 0.10],
#  [0.62, 0.58, 1.00, 0.15],
#  [0.12, 0.10, 0.15, 1.00]]
```

---

### 2. Semantic Search

```python
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('intfloat/multilingual-e5-large')

# Dokumente (Produktbeschreibungen)
corpus = [
    "passage: Der Kirsch LABO-288 ist ein Laborkühlschrank mit 280L Volumen",
    "passage: Liebherr HMF-4001 Medikamentenkühlschrank nach DIN 13277",
    "passage: Pommes Frites zubereiten in 10 Minuten",
    "passage: Gefrierschrank für Labor mit -40°C",
]

# Corpus embedden
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Query
query = "query: Kühlschrank für Medikamente mit DIN-Zertifizierung"
query_embedding = model.encode(query, convert_to_tensor=True)

# Suche
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]

print(f"\nQuery: {query}\n")
for hit in hits:
    print(f"Score: {hit['score']:.4f} - {corpus[hit['corpus_id']]}")

# Output:
# Score: 0.7834 - passage: Liebherr HMF-4001 Medikamentenkühlschrank nach DIN 13277
# Score: 0.6521 - passage: Der Kirsch LABO-288 ist ein Laborkühlschrank mit 280L Volumen
# Score: 0.2341 - passage: Gefrierschrank für Labor mit -40°C
```

---

### 3. Batch Processing (große Datenmengen)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Große Anzahl Texte
texts = ["Text " + str(i) for i in range(10000)]

# Batch-Encoding für Effizienz
embeddings = model.encode(
    texts,
    batch_size=32,        # 32 Texte parallel
    show_progress_bar=True,
    convert_to_numpy=True
)

print(embeddings.shape)  # (10000, 384)

# Speichern
np.save('embeddings.npy', embeddings)

# Laden
embeddings_loaded = np.load('embeddings.npy')
```

---

### 4. Mit Normalisierung

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = ["Kühlschrank", "Refrigerator"]
embeddings = model.encode(texts)

# Normalisieren (L2-Norm)
embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Prüfen
print(np.linalg.norm(embeddings_normalized[0]))  # 1.0
print(np.linalg.norm(embeddings_normalized[1]))  # 1.0

# Jetzt: Cosine = Dot Product
cosine = np.dot(embeddings_normalized[0], embeddings_normalized[1])
print(f"Similarity: {cosine:.4f}")
```

---

### 5. Custom Pooling

```python
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('bert-base-uncased')

def custom_pooling(token_embeddings, attention_mask):
    # Gewichte erste Tokens höher (oft wichtiger!)
    seq_len = token_embeddings.size(1)
    weights = torch.linspace(1.0, 0.5, seq_len).to(token_embeddings.device)

    # Gewichteter Durchschnitt
    weighted = token_embeddings * weights.unsqueeze(0).unsqueeze(-1)
    sum_embeddings = torch.sum(weighted * attention_mask.unsqueeze(-1), dim=1)
    sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)

    return sum_embeddings / sum_mask

# Nutze Custom Pooling
# (siehe Sentence-Transformers Docs für Integration)
```

---

## 🚀 Was du danach kannst

**Architektur-Verständnis:**
- Du verstehst die Transformer-Architektur und Self-Attention Mechanismus
- Du erkennst den Unterschied zwischen BERT (Token-Level) und Sentence-BERT
- Du verstehst verschiedene Pooling-Strategien und ihre Anwendungsfälle

**Praktische Implementierung:**
- Du wählst das richtige Dense Embedding Model für deinen Use Case
- Du implementierst effiziente Batch-Processing für große Datenmengen
- Du optimierst Model-Performance für deutsche und englische Texte

**Production-Skills:**
- Du verstehst Training-Methoden (Contrastive, Triplet, Multiple Negatives)
- Du integrierst Dense Embeddings in RAG-Systeme
- Du kombinierst Dense mit Sparse Embeddings für optimale Retrieval-Performance

### Model-Empfehlungen nach Use Case:
- **Deutsch + schnell:** `multilingual-e5-small`
- **Deutsch + genau:** `multilingual-e5-large`
- **Englisch + schnell:** `all-MiniLM-L6-v2`
- **Englisch + genau:** `all-mpnet-base-v2`

## 🔗 Weiterführende Themen

**Verwandte Kapitel:**
- [03-sentence-embeddings.md](03-sentence-embeddings.md) - Sparse Embeddings als Alternative
- [04-dense-vs-sparse.md](04-dense-vs-sparse.md) - Dense + Sparse Hybrid-Strategien
- [08-model-selection.md](08-model-selection.md) - Systematische Model-Evaluation

**Papers & Ressourcen:**
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. 2017 (Transformer)
- [BERT Paper](https://arxiv.org/abs/1810.04805) - Devlin et al. 2018
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Reimers & Gurevych 2019

**Code & Tools:**
- [Sentence-Transformers Library](https://www.sbert.net/) - Production-ready Implementation
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Model Hub
