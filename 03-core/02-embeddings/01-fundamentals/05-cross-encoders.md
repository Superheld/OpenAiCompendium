# Cross-Encoders: Deep Dive

## Inhaltsverzeichnis

1. [Was sind Cross-Encoders?](#was-sind-cross-encoders)
2. [Bi-Encoder vs. Cross-Encoder](#bi-encoder-vs-cross-encoder)
3. [Wie funktioniert ein Cross-Encoder?](#wie-funktioniert-ein-cross-encoder)
4. [Training und Fine-Tuning](#training-und-fine-tuning)
5. [Re-Ranking Pipeline](#re-ranking-pipeline)
6. [Vor- und Nachteile](#vor--und-nachteile)
7. [Use-Cases und Beispiele](#use-cases-und-beispiele)

---

## Was sind Cross-Encoders?

### Die präziseste Methode für Text-Matching

**Cross-Encoder** ist eigentlich **kein Embedding-Verfahren** im klassischen Sinne. Stattdessen berechnet er direkt einen **Relevanz-Score** zwischen zwei Texten.

### Der fundamentale Unterschied

**Bei Embeddings (Bi-Encoder):**
```
Text A → Vektor A [0.23, -0.45, ...]
Text B → Vektor B [0.12, 0.34, ...]

Vergleich: Cosine(A, B) = 0.75
```

**Bei Cross-Encoder:**
```
Text A + Text B → BERT → Relevanz-Score = 0.82

Kein Vektor! Direkt ein Score.
```

### Warum "Cross"?

Der Begriff kommt von der **Cross-Attention** - beide Texte "sehen" sich gegenseitig während des Encodings.

```
Standard (Bi-Encoder):
  Text A: "Kühlschrank"
  ↓ Encoding (isoliert)
  Vector A

  Text B: "Refrigerator"
  ↓ Encoding (isoliert)
  Vector B

  A weiß nichts über B beim Encoding!

Cross-Encoder:
  "Kühlschrank [SEP] Refrigerator"
  ↓ Encoding (zusammen!)

  A und B interagieren im BERT!
  → "Kühlschrank" sieht "Refrigerator"
  → Attention lernt: Das sind Synonyme!
```

### Kernidee

**Maximum an Information durch Joint Encoding:**

Wenn Query und Dokument **zusammen** durch ein Transformer-Model gehen, kann das Model:
- **Synonyme erkennen** ("Kühlschrank" ↔ "Refrigerator")
- **Widersprüche finden** ("billig" vs. "teuer")
- **Kontext verstehen** ("Bank" am Fluss vs. Geldinstitut)
- **Semantische Beziehungen** erfassen (Ursache-Wirkung, Teil-Ganzes)

**Der Preis:** Sehr langsam, weil jedes Query-Doc-Paar neu berechnet werden muss.

---

## Bi-Encoder vs. Cross-Encoder

### Architektur-Vergleich

**Bi-Encoder (Standard Dense Embeddings):**

```
Query: "Laborkühlschrank 280L"
    ↓
  [BERT]
    ↓
  [Vector] [0.23, -0.45, 0.12, ...]

Document: "Medizinischer Kühlschrank 280 Liter"
    ↓
  [BERT] ← Separates Encoding!
    ↓
  [Vector] [0.25, -0.43, 0.14, ...]

Cosine Similarity = 0.87
```

**Eigenschaften:**
- Dokumente können **vorher** encoded werden (einmal!)
- Query wird encoded (einmal!)
- Vergleich ist Vektor-Operation (sehr schnell)
- Skaliert zu Millionen Dokumenten

---

**Cross-Encoder:**

```
Query + Document zusammen:
"[CLS] Laborkühlschrank 280L [SEP] Medizinischer Kühlschrank 280 Liter [SEP]"
    ↓
  [BERT mit Cross-Attention]
    ↓
  [CLS]-Token Embedding
    ↓
  [Linear Layer]
    ↓
  Score: 0.92
```

**Eigenschaften:**
- Dokumente können **nicht** vorher encoded werden
- Für jede neue Query müssen **alle** Docs neu berechnet werden
- Vergleich ist komplettes BERT-Forward-Pass (langsam)
- Skaliert schwer (nur für Re-Ranking von wenigen Kandidaten)

### Warum ist Cross-Encoder genauer?

**1. Token-zu-Token Attention**

Bei Bi-Encoder:
```
"Laborkühlschrank" wird encoded ohne "280 Liter" zu sehen
→ Embedding ist generisch für alle Kühlschränke
```

Bei Cross-Encoder:
```
"Laborkühlschrank" sieht "280 Liter" während Encoding
→ Attention kann fokussieren: "Ah, es geht um Volumen!"
→ Spezialisiertes Verständnis
```

**2. Asymmetrische Beziehungen**

Bi-Encoder: Symmetrisch
```
Sim(A, B) = Sim(B, A)
Cosine ist immer symmetrisch
```

Cross-Encoder: Kann asymmetrisch sein
```
Query: "Was ist ein Kühlschrank?"
Doc: "Ein Kühlschrank ist ein Gerät..."

vs.

Query: "Ein Kühlschrank ist ein Gerät..."
Doc: "Was ist ein Kühlschrank?"

Cross-Encoder kann lernen: Erste Kombination ist relevant, zweite nicht!
```

**3. Negative Signale**

Bi-Encoder:
```
Query: "Kühlschrank OHNE Gefrierfach"
Doc: "Kühlschrank mit Gefrierfach"

Embedding von "OHNE" und "mit" sind beide ähnlich (beide Präpositionen)
→ Hohe Similarity trotz Widerspruch!
```

Cross-Encoder:
```
Sieht "OHNE" und "mit" zusammen
→ Attention erkennt Widerspruch
→ Niedriger Score!
```

### Performance-Unterschied

**Empirisch (BEIR Benchmark):**

```
Task: Information Retrieval

Bi-Encoder (Dense):
  nDCG@10 = 0.52

Cross-Encoder (Re-Ranking):
  nDCG@10 = 0.61  ← +17% Verbesserung!

Aber:
  Bi-Encoder: 1M Docs in 100ms
  Cross-Encoder: 100 Docs in 2000ms
```

**Trade-off:** Genauigkeit vs. Geschwindigkeit

---

## Wie funktioniert ein Cross-Encoder?

### Architektur im Detail

**Input-Konstruktion:**

```
Query: "Laborkühlschrank DIN 13277"
Document: "Der Kirsch LABO-288 ist nach DIN 13277 zertifiziert"

Combined Input:
[CLS] Laborkühlschrank DIN 13277 [SEP] Der Kirsch LABO-288 ist nach DIN 13277 zertifiziert [SEP]
  ↑                               ↑                                                          ↑
Special  Query Tokens            Separator        Document Tokens                     Separator
```

**Wichtig:**
- `[CLS]` - Classification Token (wird für finalen Score genutzt)
- Erste `[SEP]` - trennt Query von Document
- Zweite `[SEP]` - markiert Ende

### BERT Processing

**Self-Attention über beide Texte:**

```
Attention Matrix (vereinfacht):

           [CLS] Labor... DIN  [SEP] Der  Kirsch LABO  ist  nach  DIN  ...
[CLS]      [0.1  0.2     0.3   0.1   0.05  0.1   0.05  0.1  0.05  0.3  ...]
Labor...   [0.1  0.4     0.2   0.05  0.1   0.2   0.35  0.1  0.1   0.15 ...] ← Attention!
DIN        [0.2  0.1     0.3   0.05  0.05  0.05  0.1   0.1  0.15  0.45 ...] ← Matched "DIN"!
[SEP]      [0.15 0.1     0.15  0.2   0.1   0.15  0.1   0.1  0.1   0.15 ...]
...
```

**Cross-Attention findet:**
- "Labor..." achtet auf "LABO" (0.35) ← Prefix-Match!
- "DIN" achtet auf "DIN" (0.45) ← Exakter Match!
- `[CLS]` aggregiert alle Informationen (0.1-0.3 über alle)

**Nach 12 Transformer-Layern:**
```
[CLS]-Token Embedding: [768 Dimensionen]
  Kodiert die Relevanz-Information!
```

### Classification Head

**Final Score Berechnung:**

```
[CLS] Embedding: [768-dim]
    ↓
Linear Layer: 768 → 1
    ↓
Sigmoid (optional)
    ↓
Score: 0.0 - 1.0
```

**Verschiedene Objectives möglich:**

**1. Binary Classification:**
```
Output: [0, 1]
  0 = nicht relevant
  1 = relevant
```

**2. Regression:**
```
Output: [0.0 - 1.0]
  Continuous score
  0.0 = völlig irrelevant
  1.0 = perfektes Match
```

**3. Multi-Class:**
```
Output: [not_relevant, somewhat_relevant, very_relevant]
  3 Klassen statt binär
```

### Token-Type Embeddings

**BERT unterscheidet Query von Document intern:**

```
Tokens:     [CLS] Labor... DIN [SEP] Der Kirsch [SEP]
Token IDs:  [101  2342    5123 102   4532 8234  102  ]
Segment:    [  0    0      0    0     1    1     1  ]
            ↑ Query Segment         ↑ Document Segment

Embedding = Token Embedding + Position Embedding + Segment Embedding
```

**Segment Embedding hilft:**
- Model weiß "Das ist die Query"
- Model weiß "Das ist das Dokument"
- Kann unterschiedlich behandeln (z.B. Query-Terms wichtiger)

---

## Training und Fine-Tuning

### Dataset-Anforderungen

**Cross-Encoder braucht:**

```
(Query, Document, Label)
```

**Beispiele:**

```
("Kühlschrank 280L", "LABO-288 Laborkühlschrank 280 Liter", 1)  ← Relevant
("Kühlschrank 280L", "Pommes Frites Rezept", 0)                ← Nicht relevant
("DIN 13277", "Medikamentenkühlschrank nach DIN 13277", 1)     ← Relevant
("DIN 13277", "Laborkühlschrank ohne Zertifizierung", 0)        ← Nicht relevant
```

**Label-Arten:**

**Binary:**
```
0 = nicht relevant
1 = relevant
```

**Graded (besser!):**
```
0 = nicht relevant
1 = etwas relevant
2 = relevant
3 = sehr relevant
```

**Continuous:**
```
0.0 - 1.0 (Similarity Score)
```

### Training-Prozess

**1. Pre-Training (optional):**

Starte mit vortrainiertem BERT:
- `bert-base-uncased` (Englisch)
- `bert-base-german-cased` (Deutsch)
- `xlm-roberta-base` (Multilingual)

**2. Fine-Tuning:**

**Loss-Function (Binary Cross-Entropy):**

```
Für jedes (Query, Doc, Label):

  Score = CrossEncoder(Query, Doc)

  Loss = -[Label × log(Score) + (1-Label) × log(1-Score)]

  Backprop → Update Weights
```

**Für Regression (MSE):**

```
Loss = (Predicted_Score - True_Score)²
```

**3. Hard Negative Mining:**

Nicht alle Negative sind gleich lehrreich!

```
Easy Negative (nutzlos):
  Query: "Laborkühlschrank"
  Doc: "Pommes Frites" ← Offensichtlich irrelevant

Hard Negative (sehr wertvoll!):
  Query: "Laborkühlschrank DIN 13277"
  Doc: "Laborkühlschrank DIN 13271" ← Fast relevant, aber falsche Norm!
```

**Mining-Strategie:**
1. Nutze Bi-Encoder für initiales Retrieval
2. Top-100 Kandidaten
3. Davon sind einige relevant, viele "hard negatives"
4. Nutze diese für Cross-Encoder Training

### Knowledge Distillation

**Problem:** Cross-Encoder sehr groß/langsam

**Lösung:** Lehrer-Schüler-Training

```
Teacher (Cross-Encoder - groß, genau):
  BERT-large mit 24 Layern

  Query + Doc → Teacher Score = 0.87

Student (Bi-Encoder - klein, schnell):
  BERT-base mit 12 Layern

  Query → Vec_Q
  Doc → Vec_D

  Similarity = Cosine(Vec_Q, Vec_D)

  Loss = (Similarity - 0.87)²  ← Imitiere Teacher!
```

**Resultat:** Bi-Encoder lernt von Cross-Encoder, wird fast so gut, aber bleibt schnell!

---

## Re-Ranking Pipeline

### Two-Stage Retrieval

**Das Standard-Pattern für Production:**

```
Stage 1: Candidate Generation (schnell, breit)
  ↓
  Bi-Encoder oder BM25
  1 Million Docs → Top-1000 Kandidaten
  Zeit: ~100ms

Stage 2: Re-Ranking (langsam, präzise)
  ↓
  Cross-Encoder
  1000 Kandidaten → Top-10 Final Results
  Zeit: ~2000ms

Total: ~2.1 Sekunden
```

**Warum funktioniert das?**

- **Recall (Stage 1):** Bi-Encoder/BM25 haben hohen Recall
  - Von 10 relevanten Docs finden sie 9 in Top-1000

- **Precision (Stage 2):** Cross-Encoder hat hohe Precision
  - Von 1000 Kandidaten findet er die besten 10

### Multi-Stage Refinement

**Für sehr große Datenmengen:**

```
Stage 1: BM25
  100M Docs → Top-10k
  Zeit: 50ms

Stage 2: Bi-Encoder
  10k → Top-100
  Zeit: 200ms

Stage 3: Cross-Encoder
  100 → Top-10
  Zeit: 500ms

Total: ~750ms
```

**Trade-off:** Mehr Stages = mehr Komplexität, aber bessere Latenz

### Hybrid Re-Ranking

**Kombiniere mehrere Signale:**

```
Für jedes Dokument:

  BM25_Score = 0.75
  Dense_Score = 0.82
  Cross_Encoder_Score = 0.89

  Final_Score = α×BM25 + β×Dense + γ×Cross_Encoder
              = 0.2×0.75 + 0.3×0.82 + 0.5×0.89
              = 0.841
```

**Parameter (α, β, γ):**
- Manuell tunen
- Grid-Search auf Validation-Set
- Oder: Learned Fusion (LambdaMART, LightGBM)

---

## Vor- und Nachteile

### Vorteile von Cross-Encoders

✅ **Höchste Genauigkeit**

Alle Benchmarks zeigen: Cross-Encoder > Bi-Encoder > BM25

```
TREC-COVID (Medical Search):
  BM25: nDCG@10 = 0.48
  Bi-Encoder: nDCG@10 = 0.56
  Cross-Encoder: nDCG@10 = 0.65  ← Beste Performance
```

✅ **Versteht Kontext besser**

```
Query: "Kühlschrank ohne Gefrierfach"
Doc A: "Kühlschrank mit separatem Gefrierfach" ← Nicht relevant!
Doc B: "Kühlschrank, kein Gefrierfach integriert" ← Relevant!

Bi-Encoder: Beide ähnlich (enthalten "Kühlschrank", "Gefrierfach")
Cross-Encoder: Erkennt "ohne" vs. "mit" → Unterschiedliche Scores
```

✅ **Semantische Relationen**

```
Query: "Ursachen von Kühlschrank-Defekten"
Doc: "Defekte Kühlschränke können durch Stromausfall entstehen"

Bi-Encoder: Moderate Similarity
Cross-Encoder: Erkennt Ursache-Wirkung-Beziehung → Hoher Score
```

✅ **Negation und Widersprüche**

```
Query: "Medikamentenkühlschrank"
Doc: "Dies ist KEIN Medikamentenkühlschrank, sondern ein Laborkühlschrank"

Bi-Encoder: Hohe Similarity (beide enthalten "Medikamentenkühlschrank")
Cross-Encoder: Niedrige Score (erkennt "KEIN")
```

---

### Nachteile von Cross-Encoders

❌ **Sehr langsam**

```
1000 Dokumente × BERT Forward Pass (50ms pro Doc)
= 50 Sekunden!

vs.

Bi-Encoder: 1000 × Cosine (0.001ms pro Doc)
= 1 Sekunde
```

❌ **Kein Pre-Computation**

```
Bi-Encoder:
  Docs können vorher embedded werden
  → Speichere Vektoren
  → Bei neuer Query nur Query embedden

Cross-Encoder:
  Jedes (Query, Doc) Paar muss neu berechnet werden
  → Keine Wiederverwendung möglich
```

❌ **Speicher-intensiv bei Inference**

```
BERT-base: 110M Parameter
→ ~440MB Model-Weights
→ Pro Forward Pass: ~2GB GPU Memory (bei Batch=32)
```

❌ **Skaliert nicht zu Millionen Docs**

```
1M Docs × 50ms = 13.9 Stunden für eine Query!

Praktisch nur für:
  - Re-Ranking (wenige Kandidaten)
  - Kleine Datenmengen (<10k Docs)
```

❌ **Keine Embeddings**

```
Bi-Encoder:
  Kann Embeddings für Clustering, Visualization nutzen

Cross-Encoder:
  Gibt nur Scores, keine Vektoren
  → Nicht für Downstream-Tasks nutzbar
```

---

### Wann Cross-Encoder nutzen?

✅ **Perfekt für:**

**1. Re-Ranking (häufigster Use-Case)**
```
Stage 1: Bi-Encoder (1M → 1000)
Stage 2: Cross-Encoder (1000 → 10)
```

**2. Kleine Datenmengen**
```
<10k Dokumente → Cross-Encoder direkt einsetzbar
Beispiel: FAQ (100 Fragen), Legal Docs (1000 Verträge)
```

**3. High-Precision Tasks**
```
Medical Search, Legal Search, Question Answering
→ Genauigkeit wichtiger als Speed
```

**4. Evaluation & Benchmarking**
```
Nutze Cross-Encoder als "Ground Truth"
→ Trainiere schnellere Models (Bi-Encoder) mit Distillation
```

❌ **Nicht geeignet für:**

**1. First-Stage Retrieval**
```
Millionen Docs → zu langsam
```

**2. Real-Time Anwendungen**
```
<100ms Latenz erforderlich → Bi-Encoder nutzen
```

**3. Embedding-basierte Tasks**
```
Clustering, Semantic Search Index, Similarity-based Recommendation
→ Braucht Vektoren, Cross-Encoder gibt keine
```

---

## Use-Cases und Beispiele

### 1. Question Answering

**Szenario:** Finde beste Antwort auf Frage

```
Frage: "Wie funktioniert ein Kühlschrank?"

Kandidaten (von Bi-Encoder):
  1. "Ein Kühlschrank ist ein Gerät zur Kühlung..."
  2. "Kühlschränke gibt es in verschiedenen Größen..."
  3. "Der Kühlkreislauf funktioniert durch Kompression..."
  4. "Laborkühlschränke unterscheiden sich von..."

Cross-Encoder Re-Ranking:
  1. Doc 3: 0.92 ← Erklärt "wie funktioniert"!
  2. Doc 1: 0.78 ← Definiert nur
  3. Doc 4: 0.45
  4. Doc 2: 0.23
```

**Cross-Encoder erkennt:** Query fragt nach "wie", nicht "was"

### 2. Duplicate Detection

**Szenario:** Finde doppelte Produktbeschreibungen

```
Doc A: "Kirsch LABO-288 Laborkühlschrank 280 Liter"
Doc B: "LABO-288 von Kirsch, Laborkühlschrank mit 280L Volumen"

Bi-Encoder: Similarity = 0.75 (nicht eindeutig)
Cross-Encoder: Score = 0.95 ← Sehr wahrscheinlich Duplikat!

Doc C: "Liebherr Laborkühlschrank 280 Liter"
Cross-Encoder: Score = 0.65 ← Ähnlich, aber anderer Hersteller
```

### 3. Sentiment-Aware Search

**Szenario:** Finde Reviews mit passender Stimmung

```
Query: "positive Erfahrungen mit Laborkühlschränken"

Doc A: "Der Kühlschrank funktioniert leider nicht gut..."
  Bi-Encoder: 0.7 (enthält "Kühlschrank", "funktioniert")
  Cross-Encoder: 0.2 ← Erkennt negative Stimmung!

Doc B: "Sehr zufrieden mit unserem neuen Laborkühlschrank..."
  Bi-Encoder: 0.72
  Cross-Encoder: 0.91 ← Erkennt positive Stimmung!
```

### 4. Multi-Hop Reasoning

**Szenario:** Komplexe Fragen die mehrere Fakten kombinieren

```
Query: "Welche DIN-zertifizierten Kühlschränke haben über 250 Liter?"

Doc A: "Der LABO-288 hat 280 Liter Volumen"
  → Enthält "280 Liter" ✓ aber nicht "DIN"

Doc B: "DIN 13277 zertifizierte Kühlschränke..."
  → Enthält "DIN" ✓ aber nicht "Liter"

Doc C: "LABO-288 ist DIN 13277 zertifiziert mit 280L"
  → Enthält BEIDE Infos ✓✓

Bi-Encoder: A=0.7, B=0.65, C=0.75 (alle ähnlich)
Cross-Encoder: A=0.5, B=0.4, C=0.95 ← Erkennt dass C beide Bedingungen erfüllt!
```

### 5. Passage Re-Ranking

**Szenario:** Finde relevantesten Textabschnitt in langem Dokument

```
Query: "Temperaturbereich von Medikamentenkühlschränken"

Dokument (3000 Wörter):
  Absatz 1: "Medikamentenkühlschränke sind wichtig..." (allgemein)
  Absatz 5: "Der Temperaturbereich liegt zwischen +2°C und +8°C..." ← Relevant!
  Absatz 12: "Installation und Wartung..." (irrelevant)

Bi-Encoder:
  Absatz 1: 0.65 (enthält "Medikamentenkühlschrank")
  Absatz 5: 0.72
  Absatz 12: 0.45

Cross-Encoder:
  Absatz 1: 0.45
  Absatz 5: 0.94 ← Exakte Antwort!
  Absatz 12: 0.15
```

### Tools und Libraries

**Sentence-Transformers:**
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = model.predict([
    ("Query", "Document 1"),
    ("Query", "Document 2")
])
# [0.85, 0.23]
```

**Hugging Face Transformers:**
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'cross-encoder/ms-marco-MiniLM-L-6-v2'
)
# Manueller Forward Pass für mehr Kontrolle
```

**Verfügbare Pre-Trained Models:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` - Schnell, Englisch
- `cross-encoder/ms-marco-electra-base` - Besser, Englisch
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` - Multilingual
- `cross-encoder/qnli-electra-base` - Question-Answering

---

## Zusammenfassung

### Key Takeaways

1. **Cross-Encoder = Joint Encoding** von Query + Document
2. **Höchste Genauigkeit** durch volle Attention zwischen Texten
3. **Sehr langsam** - jedes Paar muss neu berechnet werden
4. **Hauptanwendung: Re-Ranking** (Stage 2 nach Bi-Encoder)
5. **Versteht Kontext** besser als Bi-Encoder (Negation, Widersprüche, Relationen)
6. **Keine Embeddings** - gibt nur Scores
7. **Two-Stage Retrieval** ist Best Practice für Production

### Architektur-Vergleich

```
┌─────────────────┬──────────────┬────────────────┬─────────────┐
│                 │ Bi-Encoder   │ Cross-Encoder  │ Multi-Vector│
├─────────────────┼──────────────┼────────────────┼─────────────┤
│ Genauigkeit     │ ⭐⭐         │ ⭐⭐⭐⭐⭐    │ ⭐⭐⭐⭐    │
│ Geschwindigkeit │ ⚡⚡⚡       │ 💤             │ ⚡⚡        │
│ Skalierbarkeit  │ 1M+ Docs     │ <10k Docs      │ 100k Docs   │
│ Pre-Compute     │ ✅           │ ❌             │ ✅          │
│ Embeddings      │ ✅           │ ❌             │ ✅          │
│ Use-Case        │ Stage 1      │ Re-Ranking     │ Stage 1/2   │
└─────────────────┴──────────────┴────────────────┴─────────────┘
```

### Production Pipeline

**Empfehlung:**
```
1. BM25 (100M → 10k) - 50ms
   ↓
2. Bi-Encoder (10k → 100) - 200ms
   ↓
3. Cross-Encoder (100 → 10) - 500ms
   ↓
Total: ~750ms für höchste Qualität
```

---

## Nächste Schritte

- [06-HYBRID-APPROACHES.md](06-HYBRID-APPROACHES.md) - Kombiniere alle Methoden optimal
- [07-QUANTIZATION.md](07-QUANTIZATION.md) - Mache Cross-Encoder schneller
- [02-DENSE-EMBEDDINGS.md](02-DENSE-EMBEDDINGS.md) - Zurück zu Bi-Encodern

---

## Weiterführende Ressourcke

**Papers:**
- Nogueira et al. 2019: "Passage Re-ranking with BERT"
- Humeau et al. 2020: "Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring"

**Datasets:**
- MS MARCO - Microsoft Machine Reading Comprehension
- Natural Questions - Google Q&A Dataset
- TREC - Text Retrieval Conference Benchmarks

**Tools:**
- Sentence-Transformers Library - Einfachste Cross-Encoder Nutzung
- Hugging Face Model Hub - Viele Pre-trained Cross-Encoders
