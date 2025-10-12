# Model Selection: Der richtige Embedding-Space für deinen Use Case

## ❓ Das Problem (Problem-First)

**Ohne richtiges Model-Verständnis geht folgendes schief:**
- **Embedding Spaces sind inkompatibel**: Du mischt Embeddings von Model A mit Model B → völlig sinnlose Similarity-Werte
- **Evaluation ist Model-abhängig**: Recall@10 mit Sentence-BERT ≠ Recall@10 mit E5 → du kannst Models nicht vergleichen
- **Falsches Model = schlechte Performance**: Englisch-only Model für deutsche Texte → Qualitätsverlust von 40%+

**Die zentrale Frage:**
Warum erzeugen verschiedene Models verschiedene "Welten" (Embedding Spaces) und wie wähle ich das richtige Model für meinen Use Case?

**Beispiel-Szenario:**
```python
# Model A
model_a = SentenceTransformer('all-MiniLM-L6-v2')
emb_a = model_a.encode("Laborkühlschrank")  # [0.23, -0.45, ..., 0.12]

# Model B
model_b = SentenceTransformer('multilingual-e5-large')
emb_b = model_b.encode("Laborkühlschrank")  # [0.89, 0.12, ..., -0.67]

# ❌ FALSCH: Similarity zwischen A und B
cosine_sim(emb_a, emb_b)  # 0.34 ← Sinnlos!
# Wie Celsius und Fahrenheit vergleichen!

# ✓ RICHTIG: Beide vom gleichen Model
emb_1 = model_a.encode("Kühlschrank")
emb_2 = model_a.encode("Gefrierschrank")
cosine_sim(emb_1, emb_2)  # 0.78 ← Sinnvoll!
```

## 🎯 Lernziele

Nach diesem Kapitel kannst du:
- [ ] Du verstehst dass jedes Model einen eigenen Embedding-Space erzeugt (nicht vergleichbar!)
- [ ] Du wählst das richtige Model basierend auf Sprache, Domain, Qualitäts-Anforderungen
- [ ] Du erkennst wann Model-Wechsel alle Embeddings neu berechnen erfordert

## 🧠 Intuition zuerst (Scaffolded Progression)

### Alltagsanalogie: Verschiedene Landkarten

**Beispiel aus dem echten Leben:**

```
Google Maps (Model A):
  Berlin → [52.52°N, 13.40°E]
  München → [48.14°N, 11.58°E]
  → Distanz: 504 km

Mittelalterliche Karte (Model B):
  Berlin → [7cm von links, 12cm von oben]
  München → [5cm von links, 18cm von oben]
  → Distanz: 6.3 cm

❌ FALSCH: Google-Koordinate mit mittelalterlicher Koordinate vergleichen!
✓ RICHTIG: Nur Punkte auf DERSELBEN Karte vergleichen
```

**Bei Embeddings genauso:**
- Jedes Model = eigene "Karte" (Embedding Space)
- Punkte auf verschiedenen Karten vergleichen = sinnlos
- Nur Punkte auf GLEICHER Karte vergleichen!

### Visualisierung: Model Spaces

```
Model A Space:              Model B Space:
    Kühl-                      Apfel
    schrank                      •
       •
                                Kühl-
    Gefrier-                   schrank
    schrank                       •
       •           VS.
                                Gefrier-
    Apfel                      schrank
       •                          •

Gleiche Objekte, aber VERSCHIEDENE Positionen!
→ Distanzen NICHT vergleichbar
```

### Die Brücke zur Mathematik

**Intuition:** Jedes Model lernt eigene Transformation Text → Vektor

**Mathematisch:**
```
Model A: f_A: Text → ℝ^384
Model B: f_B: Text → ℝ^1024

f_A("Kühlschrank") ≠ f_B("Kühlschrank")
→ Verschiedene Vektorräume!
→ Verschiedene Dimensionen!
→ Verschiedene "Bedeutung" pro Dimension!
```

**Warum?**
- Training-Daten unterschiedlich
- Architektur unterschiedlich
- Objektive (Loss) unterschiedlich
→ Gelernte Repräsentation unterschiedlich!

## 🧮 Das Konzept verstehen

### 1. Embedding Spaces sind Model-spezifisch

#### Was ist ein Embedding Space?

**Definition:**
Der hochdimensionale Raum den ein Model "gelernt" hat, um Bedeutung zu repräsentieren.

```python
# Model A lernt einen Space:
model_a = SentenceTransformer('all-MiniLM-L6-v2')
# → Space: ℝ^384
# → Training: NLI + Paraphrase Datasets
# → Objective: Contrastive Loss

# Model B lernt ANDEREN Space:
model_b = SentenceTransformer('multilingual-e5-large')
# → Space: ℝ^1024
# → Training: 1B multilingual pairs
# → Objective: Multiple Negatives Ranking Loss

# VERSCHIEDENE Spaces!
```

#### Warum sind Spaces inkompatibel?

**Grund 1: Verschiedene Dimensionen**
```python
model_a.encode("text").shape  # (384,)
model_b.encode("text").shape  # (1024,)

# Kann man nicht mal dot product machen!
```

**Grund 2: Selbst bei gleicher Dimension - verschiedene "Bedeutung"**
```python
# Beide 768-dim, aber:
model_1 = SentenceTransformer('bert-base')
model_2 = SentenceTransformer('roberta-base')

emb_1 = model_1.encode("Kühlschrank")  # [0.1, 0.2, ..., 0.3]
emb_2 = model_2.encode("Kühlschrank")  # [0.9, -0.5, ..., 0.1]

# Dimension 0 bedeutet bei Model 1 vielleicht "Formalität"
# Dimension 0 bedeutet bei Model 2 vielleicht "Sentiment"
# → Völlig verschiedene semantische Räume!
```

#### Implikation für Vector Databases

**Kritisch wichtig:**
```python
# Vector DB mit Model A befüllt
vectordb = ChromaDB()
for doc in corpus:
    emb = model_a.encode(doc)
    vectordb.add(doc, emb)

# ❌ FALSCH: Query mit Model B
query_emb = model_b.encode(query)
results = vectordb.search(query_emb)  # Garbage results!

# ✓ RICHTIG: Query mit gleichem Model
query_emb = model_a.encode(query)
results = vectordb.search(query_emb)  # Sinnvolle Results!
```

**Regel:**
```
Ein Embedding Space pro Vector Database.
Model-Wechsel = Alle Embeddings neu berechnen.
```

### 2. Evaluation ist Model-abhängig

#### Das Problem

**Verschiedene Models → verschiedene Retrieval-Qualität:**

```python
# Test-Set: 1000 Query-Doc Pairs
queries = ["Kühlschrank 280L", ...]
relevant_docs = [["Doc_123", "Doc_456"], ...]

# Model A
model_a = SentenceTransformer('all-MiniLM-L6-v2')
recall_a = compute_recall_at_10(model_a, queries, relevant_docs)
# Recall@10 = 0.75

# Model B
model_b = SentenceTransformer('multilingual-e5-large')
recall_b = compute_recall_at_10(model_b, queries, relevant_docs)
# Recall@10 = 0.82

# Model B ist besser für diesen Use Case!
```

**Warum unterschiedlich?**
- Model B lernte bessere Repräsentation für diese Domain
- Model B ist multilingual → generalisiert besser
- Model B größer (1024 vs 384 dims) → mehr Kapazität

#### MTEB Benchmark verstehen

**Massive Text Embedding Benchmark:**

| Model | Avg Score | Retrieval | Classification | Clustering |
|-------|-----------|-----------|----------------|------------|
| **E5-large-v2** | 66.1 | 52.8 | 75.0 | 44.5 |
| **all-mpnet-base-v2** | 63.3 | 50.2 | 73.8 | 43.7 |
| **all-MiniLM-L6-v2** | 58.8 | 41.9 | 66.8 | 42.4 |

**Interpretation:**
- **Avg Score**: Durchschnitt über 56 Datasets
- **Retrieval**: Relevant für RAG/Search
- **Classification**: Relevant für Categorization
- **Clustering**: Relevant für Topic Modeling

**Wichtig:**
```
MTEB-Score ist NICHT absolut!
→ Dein Use Case kann anders performen
→ Immer auf eigenem Dataset evaluieren!
```

### 3. Model-Auswahl Kriterien

#### Kriterium 1: Sprache

```python
# Englisch-only Use Case:
model = SentenceTransformer('all-mpnet-base-v2')  # ✓

# Deutsch/Multilingual:
model = SentenceTransformer('all-mpnet-base-v2')  # ❌ Schlecht!
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # ✓

# Test:
emb_en = model.encode("refrigerator")
emb_de = model.encode("Kühlschrank")
similarity = cosine_sim(emb_en, emb_de)

# Englisch-only Model: 0.3 (schlecht!)
# Multilingual Model: 0.85 (gut!)
```

#### Kriterium 2: Domain

```python
# General Domain (Wikipedia, News):
model = SentenceTransformer('all-mpnet-base-v2')  # ✓

# Specific Domain:
# Legal:
model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')

# Medical:
model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')

# Scientific Papers:
model = SentenceTransformer('allenai/specter2')

# Performance-Unterschied: 10-30% bei Domain-spezifischen Tasks!
```

#### Kriterium 3: Qualität vs. Speed

| Model | Dims | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **MiniLM-L6** | 384 | ⚡⚡⚡ | ⭐⭐⭐ | Prototyping, Hobby |
| **MPNet-base** | 768 | ⚡⚡ | ⭐⭐⭐⭐ | Production (balanced) |
| **E5-large** | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | High-quality required |

```python
# Benchmark auf deinem Corpus:
import time

models = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'intfloat/multilingual-e5-large'
]

for model_name in models:
    model = SentenceTransformer(model_name)

    start = time.time()
    embeddings = model.encode(corpus[:1000])
    duration = time.time() - start

    recall = evaluate_retrieval(model, test_queries)

    print(f"{model_name}:")
    print(f"  Time: {duration:.2f}s")
    print(f"  Recall@10: {recall:.3f}")
```

## ⚠️ Häufige Missverständnisse

### ❌ Missverständnis 1: "Ich kann Models mischen"

**Warum falsch:**
```python
# 50% der Docs mit Model A embedded
vectordb.add_batch(docs[:500], model_a.encode(docs[:500]))

# 50% mit Model B embedded
vectordb.add_batch(docs[500:], model_b.encode(docs[500:]))

# Query mit Model A
query_emb = model_a.encode(query)
results = vectordb.search(query_emb, k=10)

# Problem: Docs[500:] haben Random-Scores!
# → Retrieval komplett kaputt
```

**✓ Richtig:**
EIN Model pro Vector Database. Bei Wechsel: Alles neu embedden.

**Merksatz:** "Ein Space, ein Model - niemals mischen!"

### ❌ Missverständnis 2: "Mehr Dimensionen = bessere Qualität"

**Warum falsch:**
```python
# Schlecht trainiertes 1024-dim Model
model_bad = SomeRandomModel(dim=1024)
# Recall@10: 0.45

# Gut trainiertes 384-dim Model
model_good = SentenceTransformer('all-MiniLM-L6-v2')
# Recall@10: 0.68

# Training-Daten + Architektur > Dimensionalität!
```

**✓ Richtig:**
Benchmark auf eigenem Dataset, nicht nur auf Dimensionen schauen.

**Merksatz:** "Qualität kommt vom Training, nicht von der Größe!"

### ❌ Missverständnis 3: "MTEB Top-1 ist immer am besten"

**Warum falsch:**
```python
# MTEB Top-1 auf English News-Corpus
model_mteb = SentenceTransformer('best-mteb-model')
# Dein Use Case: Deutsche Medizin-Texte

# Performance:
recall_mteb = 0.52  # Schlecht!

# Domain-spezifisches Model:
model_medical_de = SentenceTransformer('german-medical-bert')
recall_medical = 0.79  # Viel besser!

# MTEB misst General Performance!
# Dein spezifischer Use Case kann anders sein!
```

**✓ Richtig:**
MTEB als Orientierung, aber immer selbst evaluieren.

**Merksatz:** "MTEB zeigt Trends, dein Dataset zeigt Wahrheit!"

## 🔬 Hands-On: Model-Auswahl praktisch

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Test-Corpus (Deutsch, Medizin-Domain)
corpus = [
    "Laborkühlschrank mit 280 Liter Volumen DIN EN 13277",
    "Medikamentenkühlschrank für Apotheken",
    "Blutkonservenkühlschrank mit Alarmfunktion",
    # ... weitere Docs
]

queries = [
    "Kühlschrank für Medikamente",
    "Labor Kühlung DIN",
    # ... weitere Queries
]

# Ground Truth: Welche Docs relevant für Query?
ground_truth = {
    0: [0, 1],  # Query 0 → Doc 0, 1 relevant
    1: [0, 2],  # Query 1 → Doc 0, 2 relevant
}

# Models zum Testen
models_to_test = [
    'all-MiniLM-L6-v2',  # Englisch-only
    'paraphrase-multilingual-MiniLM-L12-v2',  # Multilingual
    'intfloat/multilingual-e5-large',  # Multilingual SOTA
]

def evaluate_model(model_name, corpus, queries, ground_truth):
    model = SentenceTransformer(model_name)

    # Embed corpus
    corpus_embs = model.encode(corpus, convert_to_tensor=True)

    # Evaluate
    recalls = []
    for q_idx, query in enumerate(queries):
        query_emb = model.encode(query, convert_to_tensor=True)

        # Search
        hits = util.semantic_search(query_emb, corpus_embs, top_k=10)[0]
        retrieved = [hit['corpus_id'] for hit in hits]

        # Recall@10
        relevant = ground_truth[q_idx]
        recall = len(set(retrieved) & set(relevant)) / len(relevant)
        recalls.append(recall)

    avg_recall = np.mean(recalls)
    return avg_recall

# Benchmark
print("Model Performance auf deutschem Medizin-Corpus:\n")
for model_name in models_to_test:
    recall = evaluate_model(model_name, corpus, queries, ground_truth)
    print(f"{model_name:50} Recall@10: {recall:.3f}")

# Erwartetes Ergebnis:
# all-MiniLM-L6-v2                             Recall@10: 0.450  (Englisch-only!)
# paraphrase-multilingual-MiniLM-L12-v2        Recall@10: 0.750  (Multilingual)
# intfloat/multilingual-e5-large               Recall@10: 0.850  (Best!)
```

## ⏱️ 5-Minuten-Experte

### 1. Frage: Kann ich Model wechseln ohne neu zu embedden?

<details><summary>💡 Antwort</summary>

**Nein, niemals!**

Embedding Spaces sind Model-spezifisch. Wechsel erfordert:
1. Alle Docs neu embedden mit neuem Model
2. Vector DB neu befüllen
3. Downtime oder Blue-Green Deployment

**Ausnahme:** Modell-Distillation wo Student explizit Teacher-Space imitiert (selten).

**Merksatz:** "Model-Wechsel = Rebuild Everything"
</details>

### 2. Frage: E5-large vs. MPNet - welches für Production?

<details><summary>💡 Antwort</summary>

**Kommt auf Requirements an:**

**E5-large wenn:**
- Multilingual wichtig
- Qualität > Latenz
- Budget für 1024-dim Vektoren

**MPNet wenn:**
- Nur Englisch
- Latenz wichtig
- Sparsamkeit (768-dim = 25% weniger Speicher)

**Benchmark beide auf eigenem Dataset!**
</details>

### 3. Frage: Domain-specific finetunen oder pretrained nehmen?

<details><summary>💡 Antwort</summary>

**Decision Tree:**

```
Hast du >10k Domain-Paare (Query-Doc)?
├─ Ja → Finetune (10-30% Improvement)
└─ Nein → Pretrained (zu wenig Daten)

Budget für Training?
├─ Ja → Finetune ab 1k Paaren mit Few-Shot
└─ Nein → Best pretrained multilingual Model
```

**Merksatz:** "Finetune wenn Daten + Budget, sonst best pretrained"
</details>

## 📊 Model-Empfehlungen

### Nach Use Case

| Use Case | Empfehlung | Warum? |
|----------|------------|--------|
| **Prototyping (Deutsch)** | `paraphrase-multilingual-MiniLM-L12-v2` | Schnell, multilingual, gut genug |
| **Production RAG (Multiingual)** | `intfloat/multilingual-e5-large` | SOTA Qualität, 100+ Sprachen |
| **Production RAG (Englisch)** | `all-mpnet-base-v2` | Balanced, battle-tested |
| **Legal (Englisch)** | `nlpaueb/legal-bert-base-uncased` | Domain-optimiert |
| **Medical (Englisch)** | `emilyalsentzer/Bio_ClinicalBERT` | PubMed trainiert |
| **Scientific Papers** | `allenai/specter2` | Citation-aware |

### Nach Constraint

| Constraint | Empfehlung | Trade-off |
|------------|------------|-----------|
| **Latenz <50ms** | `all-MiniLM-L6-v2` | -10% Qualität |
| **Speicher <2GB** | `all-MiniLM-L6-v2` | Kleinere Embeddings |
| **Mobile/Edge** | `all-MiniLM-L6-v2` + Quantization | -15% Qualität |
| **Maximale Qualität** | `intfloat/multilingual-e5-large` | 3x langsamer |

## 🛠️ Tools für Model Selection

```python
# MTEB Benchmark
from mteb import MTEB

model = SentenceTransformer('your-model')
evaluation = MTEB(tasks=["Retrieval"])
results = evaluation.run(model)

# Eigenes Benchmark
def benchmark_on_your_data(model_names, corpus, queries, ground_truth):
    results = {}
    for name in model_names:
        model = SentenceTransformer(name)
        recall = evaluate_retrieval(model, corpus, queries, ground_truth)
        latency = measure_encoding_time(model, corpus[:100])
        results[name] = {"recall": recall, "latency": latency}
    return results
```

## 🚀 Was du jetzt kannst

**Verständnis:**
- ✓ Du verstehst dass Embedding Spaces Model-spezifisch und inkompatibel sind
- ✓ Du erkennst dass Evaluation Model-abhängig ist
- ✓ Du verstehst MTEB als Orientierung, nicht als absolute Wahrheit

**Praktische Fähigkeiten:**
- ✓ Du benchmarkst verschiedene Models auf eigenem Dataset
- ✓ Du wählst Model basierend auf Sprache, Domain, Constraints
- ✓ Du planst Model-Migration (alle Embeddings neu)

**Kritisches Denken:**
- ✓ Du mischt niemals Embeddings verschiedener Models
- ✓ Du evaluierst vor Production-Deployment
- ✓ Du verstehst Trade-offs (Qualität vs. Speed vs. Speicher)

## 🔗 Weiterführende Themen

**Nächster logischer Schritt:**
→ [04-vector-databases.md](04-vector-databases.md) - Vector DBs, Quantization, Deployment Patterns
→ [../../04-advanced/02-retrieval-optimization.md](../../04-advanced/02-retrieval-optimization.md) - Chunking Strategies, Two-Stage Retrieval, Hybrid Search

**Praktische Anwendung:**
→ [../../06-applications/rag-systems.md](../../06-applications/rag-systems.md) - Vollständiges RAG-System mit Model Selection

**Verwandte Konzepte:**
- [../evaluation/metrics.md](../evaluation/metrics.md) - Recall@k, MRR, nDCG für Retrieval-Qualität
- [../training/contrastive-learning.md](../training/contrastive-learning.md) - Wie Embedding Models trainiert werden
