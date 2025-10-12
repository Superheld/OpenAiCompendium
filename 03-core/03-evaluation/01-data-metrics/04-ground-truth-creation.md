# Ground Truth Creation: Test-Daten für Retrieval-Evaluation aufbereiten

## ❓ Das Problem (Problem-First)

**Ohne Ground Truth kannst du nicht evaluieren:**

1. **Du weißt nicht ob dein System gut ist** - Keine Basis für Precision@K, Recall, MRR
   - *Beispiel:* System liefert 10 Docs, aber du weißt nicht welche davon wirklich relevant sind → Keine Metrik-Berechnung möglich

2. **Du kannst Systeme nicht vergleichen** - Ist neues Embedding-Model besser? Ohne GT: Unmöglich zu sagen
   - *Beispiel:* Sentence-Transformers vs. OpenAI - beide liefern unterschiedliche Results, aber welche sind besser?

3. **Du optimierst blind** - Änderst Chunk-Size, Prompt, Threshold - aber keine Ahnung ob es besser wird
   - *Beispiel:* Chunk-Size von 500 auf 1000 erhöht → Precision steigt oder sinkt? Ohne GT kannst du es nicht messen

**Die zentrale Herausforderung:**
Für jede Test-Query: Welche Dokumente/Chunks sind wirklich relevant? Das ist deine **Ground Truth** (GT).

**Beispiel-Szenario:**
```
Query: "Wie hoch ist der Energieverbrauch des HMFvh 4001?"

Deine Datenbank: 500 Chunks

Ground Truth (was du brauchst):
✅ Chunk 42: "HMFvh 4001: 172 kWh/Jahr" (relevant!)
✅ Chunk 156: "Energieeffizienz HMFvh 4001: A+" (relevant!)
❌ Chunk 23: "HMFvh 5000: 200 kWh/Jahr" (falsches Modell!)
❌ Chunk 89: "Türmaterial: Glas" (nicht zur Query relevant!)

→ Ohne diese Labels kannst du Precision@K nicht berechnen!
```

## 🎯 Lernziele

Nach diesem Kapitel kannst du:
- [ ] Verstehe warum Ground Truth für Evaluation essentiell ist
- [ ] Erstelle Test-Query-Sets (Anzahl, Typ, Qualität)
- [ ] Annotiere relevante Dokumente (Binary vs. Graded Relevance)
- [ ] Nutze LLMs zur GT-Generierung (mit Review-Prozess)
- [ ] Bewerte GT-Qualität (Inter-Annotator Agreement, Coverage)
- [ ] Vermeide häufige Fehler bei GT-Erstellung

## 🧠 Intuition zuerst

### Alltagsanalogie: Die Prüfung

**Stell dir vor, du entwickelst einen Such-Algorithmus:**

Du bist wie ein Lehrer, der eine Prüfung erstellt:
- **Test-Queries** = Prüfungsfragen
- **Ground Truth** = Musterlösung
- **Dein System** = Der Schüler

**Ohne Musterlösung:**
- Du kannst keine Punkte vergeben
- Du weißt nicht ob Antwort richtig ist
- Schüler bekommt kein Feedback

**Mit Musterlösung:**
- Klare Bewertungskriterien
- Objektive Punktevergabe
- Vergleichbarkeit zwischen Schülern

### Die Brücke zur Evaluation

Ground Truth ist deine "Musterlösung" für Retrieval:
- **Test-Query** = "Wie hoch ist der Energieverbrauch?"
- **Ground Truth** = Liste der relevanten Chunk-IDs: {42, 156, 203}
- **System-Output** = Liste der gefundenen Chunk-IDs: {42, 89, 156, 23, 301}
- **Evaluation** = Vergleiche System-Output mit Ground Truth → Precision@5 = 2/5 = 40%

## 🧮 Ground Truth erstellen: Der Prozess

### 1. Test-Query-Set designen

**Wie viele Queries?**

| Use Case | Min. Queries | Empfohlen | Warum? |
|----------|--------------|-----------|--------|
| Quick Check | 10-20 | 30 | Erste Einschätzung |
| Development | 30-50 | 100 | Iterative Verbesserung |
| Production Validation | 100+ | 200-500 | Statistische Signifikanz |
| Research / Benchmark | 500+ | 1000+ | Robuste Evaluation |

**Formel für Konfidenzintervall:**
```
Sample Size = (Z² × p × (1-p)) / E²

Für 95% Confidence (Z=1.96), p=0.5 (worst case), E=0.05 (5% margin):
Sample Size = (1.96² × 0.5 × 0.5) / 0.05² ≈ 384 Queries

→ ~400 Queries für robuste Production-Evaluation
```

**Aber:** Weniger ist okay für Development! Start mit 30-50.

**Query-Typen diversifizieren:**

```python
query_types = {
    'factual': [
        "Wie hoch ist der Energieverbrauch?",
        "Welche Maße hat das Gerät?"
    ],
    'comparative': [
        "Was ist der Unterschied zwischen Model A und B?",
        "Ist Model X energieeffizienter als Y?"
    ],
    'exploratory': [
        "Welche Sicherheitsfeatures gibt es?",
        "Was sind die technischen Spezifikationen?"
    ],
    'edge_cases': [
        "Wie viel kostet?",  # Unvollständig
        "energieverbrauch hmfvh",  # Typo
        "power consumption HMFvh 4001"  # Englisch
    ]
}
```

**Wichtig:**
- ✅ Echte User-Queries (nicht synthetisch erfunden)
- ✅ Verschiedene Schwierigkeitsgrade
- ✅ Edge Cases einbeziehen
- ❌ Nicht nur "einfache" Queries

### 2. Relevanz-Skala definieren

**Option A: Binary Relevance** (einfach, schnell)

```python
relevance_binary = {
    0: "Irrelevant",  # ❌ Beantwortet Query nicht
    1: "Relevant"     # ✅ Enthält Antwort auf Query
}
```

**Wann nutzen:**
- ✅ Schnelle GT-Erstellung
- ✅ Klare Relevanz (entweder/oder)
- ✅ Für Precision@K, Recall@K, MRR

**Option B: Graded Relevance** (präziser, aufwändiger)

```python
relevance_graded = {
    0: "Irrelevant",           # ❌ Nichts zur Query
    1: "Marginal",             # 🤔 Tangential relevant
    2: "Relevant",             # ✅ Beantwortet Query
    3: "Highly Relevant"       # ✅✅ Perfekte Antwort
}
```

**Wann nutzen:**
- ✅ Für NDCG@K (braucht Grades!)
- ✅ Wenn Qualitätsunterschiede wichtig sind
- ✅ Production-System mit Ranking-Optimierung

**Entscheidungshilfe:**

```
Brauchst du NDCG@K?
├─ JA → Graded Relevance (0-3)
└─ NEIN → Binary Relevance (0-1)
    └─ Schneller und ausreichend für Precision@K, MRR
```

### 3. Annotation durchführen

**Manuelle Annotation:**

```python
# Beispiel Annotation-Format
ground_truth = {
    "query_001": {
        "text": "Wie hoch ist der Energieverbrauch des HMFvh 4001?",
        "relevant_chunks": {
            42: 3,    # Chunk-ID: Relevance-Score (0-3)
            156: 2,
            203: 2
        }
    },
    "query_002": {
        "text": "Welche Sicherheitsfeatures hat das Gerät?",
        "relevant_chunks": {
            89: 3,
            91: 3,
            120: 2,
            145: 1
        }
    }
}
```

**Best Practices für Annotation:**

1. **Klare Guidelines schreiben:**
```markdown
# Annotation Guidelines

## Relevance 3 (Highly Relevant):
- Beantwortet Query vollständig
- Enthält exakte Fakten
- Beispiel: Query "Energieverbrauch?" → "172 kWh/Jahr"

## Relevance 2 (Relevant):
- Beantwortet Query teilweise
- Hilfreiche aber unvollständige Info
- Beispiel: Query "Energieverbrauch?" → "Energieeffizienzklasse A+"

## Relevance 1 (Marginal):
- Tangential relevant
- Kontext aber keine direkte Antwort
- Beispiel: Query "Energieverbrauch?" → "Gerät mit Energiesparfunktion"

## Relevance 0 (Irrelevant):
- Keine Verbindung zur Query
- Beispiel: Query "Energieverbrauch?" → "Türmaterial: Glas"
```

2. **Multiple Annotators (wenn möglich):**
```python
# Inter-Annotator Agreement prüfen
from sklearn.metrics import cohen_kappa_score

annotator_1 = [3, 2, 0, 1, 3, 2]  # Chunk-Relevanz Scores
annotator_2 = [3, 2, 1, 1, 2, 2]  # Andere Person

kappa = cohen_kappa_score(annotator_1, annotator_2)
print(f"Cohen's Kappa: {kappa:.3f}")

# Interpretation:
# Kappa > 0.8: Excellent agreement
# Kappa 0.6-0.8: Substantial agreement
# Kappa 0.4-0.6: Moderate agreement
# Kappa < 0.4: Poor agreement (Guidelines verbessern!)
```

3. **Sampling-Strategie:**
```python
# Nicht alle Chunks annotieren!
# Smart Sampling: Fokus auf wahrscheinlich relevante

def sample_chunks_for_annotation(query, all_chunks, initial_retrieval):
    """
    Strategie: Annotiere Top-50 von Initial Retrieval + 50 Random
    → Spart Zeit, behält Qualität
    """
    top_candidates = initial_retrieval[:50]  # Von BM25 oder Simple Embedding
    random_sample = random.sample(all_chunks, 50)

    to_annotate = set(top_candidates) | set(random_sample)
    return list(to_annotate)
```

### 4. LLM-basierte GT-Generierung (Dein Ansatz!)

**Workflow: LLM generiert, Mensch reviewt**

```python
# Schritt 1: LLM generiert Queries + relevante Chunks
llm_prompt = """
Du bist Annotation-Assistent.

Dokument-Chunks:
{chunks}

Aufgabe:
1. Generiere 5 realistische User-Queries zu diesen Chunks
2. Für jede Query: Markiere relevante Chunk-IDs (0-3 Scale)

Output-Format:
{
  "query": "...",
  "relevant_chunks": {
    chunk_id: relevance_score
  }
}
"""

# Schritt 2: LLM Output
llm_generated_gt = llm.generate(llm_prompt, chunks=your_chunks)

# Schritt 3: MANUELLES REVIEW (kritisch!)
def review_llm_gt(llm_gt):
    """
    Prüfe:
    - Sind Queries realistisch?
    - Sind Relevance-Scores korrekt?
    - Fehlen relevante Chunks?
    - Sind irrelevante als relevant markiert?
    """
    reviewed_gt = {}

    for query_id, query_data in llm_gt.items():
        print(f"\nQuery: {query_data['text']}")

        for chunk_id, relevance in query_data['relevant_chunks'].items():
            chunk_text = get_chunk(chunk_id)
            print(f"  Chunk {chunk_id} (Score: {relevance}): {chunk_text[:100]}...")

            # User review
            correct = input("Korrekt? (y/n/new_score): ")
            if correct == 'n':
                continue  # Skip diesen Chunk
            elif correct.isdigit():
                relevance = int(correct)  # Override Score

            reviewed_gt[query_id] = query_data

    return reviewed_gt
```

**Vorteile LLM-basiert:**
- ✅ Schnell (10x schneller als manuell)
- ✅ Skalierbar (1000+ Queries in Stunden)
- ✅ Consistent (gleiche Guidelines immer)

**Nachteile:**
- ⚠️ Muss reviewed werden! (LLM macht Fehler)
- ⚠️ Kann Edge Cases übersehen
- ⚠️ Synthetische Queries ≠ echte User-Queries

**Dein Workflow (wie du sagst):**
```
1. LLM generiert Queries + findet relevante Docs
2. Du reviewst und korrigierst:
   - Füge Antworten hinzu wo LLM sie nicht fand
   - Schreibe neue Queries wenn keine Antwort existiert
3. Finale GT ist human-validated
```

**Das ist ein guter Hybrid-Ansatz! ✅**

### 5. GT-Qualität bewerten

**Checkliste:**

```python
def evaluate_ground_truth_quality(gt_dataset):
    """
    Prüfe ob deine Ground Truth production-ready ist
    """
    checks = {}

    # 1. Genug Queries?
    num_queries = len(gt_dataset)
    checks['sufficient_queries'] = num_queries >= 50  # Minimum für Development

    # 2. Queries diversifiziert?
    query_lengths = [len(q['text'].split()) for q in gt_dataset.values()]
    checks['diverse_lengths'] = np.std(query_lengths) > 2  # Nicht alle gleich lang

    # 3. Relevante Chunks pro Query?
    relevant_per_query = [len(q['relevant_chunks']) for q in gt_dataset.values()]
    checks['has_relevant'] = all(r > 0 for r in relevant_per_query)  # Jede Query hat mind. 1
    checks['not_too_many'] = all(r < 50 for r in relevant_per_query)  # Nicht zu viele

    # 4. Relevance Distribution?
    all_scores = []
    for q in gt_dataset.values():
        all_scores.extend(q['relevant_chunks'].values())

    score_dist = {s: all_scores.count(s) for s in [0, 1, 2, 3]}
    checks['balanced_relevance'] = score_dist[3] > 0  # Hat Highly Relevant

    # 5. Edge Cases?
    short_queries = [q for q in gt_dataset.values() if len(q['text'].split()) < 3]
    checks['has_edge_cases'] = len(short_queries) > 0

    return checks

# Test
quality = evaluate_ground_truth_quality(your_gt)
print(f"GT Quality Score: {sum(quality.values())}/{len(quality)}")
```

## ⚠️ Häufige Missverständnisse (Misconception Debugging)

### ❌ Missverständnis 1: "Mehr Queries ist immer besser"

**Warum das falsch ist:**

```python
# 1000 synthetische Queries vs. 50 echte User-Queries
synthetic_gt = generate_queries_llm(num=1000)  # Schnell, aber nicht realistisch
real_gt = collect_user_queries(num=50)  # Langsam, aber realistisch

# System optimiert auf synthetic_gt:
precision_synthetic = 0.85  # Sieht gut aus!
precision_real = 0.42  # Versagt bei echten Queries!
```

**✓ Richtig ist:**
- **Qualität > Quantität**
- 50 echte User-Queries > 1000 synthetische
- Lieber klein anfangen und iterativ erweitern

**Merksatz:** "10 repräsentative Queries schlagen 100 unrealistische"

### ❌ Missverständnis 2: "LLM-generierte GT ist perfekt"

**Warum das falsch ist:**

```python
# LLM übersieht subtile Relevanz
Query: "Energieverbrauch HMFvh 4001"

LLM markiert:
✅ Chunk 42: "HMFvh 4001: 172 kWh/Jahr" (Score: 3)
❌ Chunk 156: "Energieeffizienzklasse A+" (Score: 0 - FEHLER!)

Mensch markiert:
✅ Chunk 42: Score 3
✅ Chunk 156: Score 2 (hilft bei Energieverbrauch-Einschätzung!)
```

**✓ Richtig ist:**
- LLM als **Assistent**, nicht als alleiniger Annotator
- **Immer manuelles Review** (besonders bei kritischen Use-Cases)
- LLM macht ~10-20% Fehler (je nach Komplexität)

**Workflow:**
```
LLM generates → Human reviews → Accept/Reject/Modify → Final GT
```

### ❌ Missverständnis 3: "Binary Relevance reicht immer"

**Warum das falsch ist:**

```python
Query: "Energieverbrauch HMFvh 4001"

Binary GT:
{
  42: 1,   # "172 kWh/Jahr" - relevant
  156: 1,  # "Energieeffizienzklasse A+" - relevant
  203: 1   # "Durchschnittlicher Verbrauch ähnlicher Geräte" - relevant
}

# Problem: Alle gleich behandelt, aber 42 ist viel besser als 203!

Graded GT:
{
  42: 3,   # Perfekte Antwort!
  156: 2,  # Relevant aber indirekt
  203: 1   # Marginal relevant
}

# NDCG kann jetzt Ranking-Qualität messen:
# System das 42 auf Platz 1 hat → NDCG 1.0
# System das 203 auf Platz 1 hat → NDCG 0.7
```

**✓ Richtig ist:**
- Binary reicht für **Precision@K, Recall, MRR**
- Graded nötig für **NDCG** und Ranking-Optimierung
- Entscheidung hängt von Metrik ab die du nutzen willst

**Siehe:** [03-ranking-metrics.md](03-ranking-metrics.md) - Welche Metrik braucht was?

## 🔬 Hands-On: GT erstellen für dein Projekt

### Schritt 1: Test-Queries sammeln

```python
# Option A: Aus Logs (beste Option!)
def extract_queries_from_logs(log_file, num_queries=50):
    """
    Echte User-Queries aus Production-Logs
    """
    with open(log_file) as f:
        logs = json.load(f)

    # Extrahiere Queries
    queries = [log['query'] for log in logs]

    # Sample diversifiziert
    # - Verschiedene Längen
    # - Verschiedene Themen
    # - Verschiedene Schwierigkeiten

    sampled = diversity_sample(queries, n=num_queries)
    return sampled

# Option B: Manuell erstellen (für Start)
test_queries = [
    "Wie hoch ist der Energieverbrauch des HMFvh 4001?",
    "Welche Maße hat das Gerät?",
    "Gibt es einen Temperaturalarm?",
    # ... 47 mehr
]

# Option C: LLM-generiert + Review (dein Ansatz)
test_queries = generate_queries_llm(your_documents, num=50)
test_queries = [review_and_edit(q) for q in test_queries]  # Dein Manual Review
```

### Schritt 2: Relevante Chunks annotieren

```python
def annotate_relevance(query, chunks, method='manual'):
    """
    Annotiere welche Chunks relevant sind
    """
    relevant_chunks = {}

    if method == 'manual':
        # Zeige Query + alle Chunks
        print(f"\nQuery: {query}")
        for chunk_id, chunk_text in chunks.items():
            print(f"\nChunk {chunk_id}: {chunk_text}")
            score = input("Relevance (0-3, or skip): ")

            if score.isdigit():
                relevant_chunks[chunk_id] = int(score)

    elif method == 'llm_assisted':
        # LLM schlägt vor
        suggested = llm_suggest_relevance(query, chunks)

        # Du reviewst
        print(f"\nQuery: {query}")
        print("LLM suggestions:")
        for chunk_id, score in suggested.items():
            print(f"  Chunk {chunk_id}: Score {score}")
            confirm = input("Accept (y/n/new_score)? ")

            if confirm == 'y':
                relevant_chunks[chunk_id] = score
            elif confirm.isdigit():
                relevant_chunks[chunk_id] = int(confirm)

    return relevant_chunks

# Für alle Queries durchführen
ground_truth = {}
for query_id, query_text in enumerate(test_queries):
    relevant = annotate_relevance(query_text, your_chunks, method='llm_assisted')
    ground_truth[f"query_{query_id:03d}"] = {
        'text': query_text,
        'relevant_chunks': relevant
    }

# Speichern
import json
with open('ground_truth.json', 'w') as f:
    json.dump(ground_truth, f, indent=2)
```

### Schritt 3: GT nutzen für Evaluation

```python
# Jetzt kannst du evaluieren!
from your_metrics import precision_at_k  # Siehe 03-ranking-metrics.md

def evaluate_retrieval_system(system, ground_truth):
    """
    Evaluiere dein Retrieval-System mit GT
    """
    results = {}

    for query_id, gt_data in ground_truth.items():
        query = gt_data['text']
        relevant_chunks = set(gt_data['relevant_chunks'].keys())

        # System liefert Ergebnisse
        retrieved = system.retrieve(query, k=10)

        # Berechne Metriken (siehe 03-ranking-metrics.md)
        results[query_id] = {
            'precision@5': precision_at_k(retrieved, relevant_chunks, k=5),
            'precision@10': precision_at_k(retrieved, relevant_chunks, k=10),
            # ... weitere Metriken
        }

    # Durchschnitt über alle Queries
    avg_precision_5 = np.mean([r['precision@5'] for r in results.values()])
    avg_precision_10 = np.mean([r['precision@10'] for r in results.values()])

    return {
        'precision@5': avg_precision_5,
        'precision@10': avg_precision_10,
        'per_query': results
    }

# Test
eval_results = evaluate_retrieval_system(my_rag_system, ground_truth)
print(f"Average Precision@5: {eval_results['precision@5']:.2%}")
```

## ⏱️ 5-Minuten-Experte

### 1. Verständnisfrage: Wie viele Queries brauche ich?

**Frage:** Ich entwickle ein RAG-System. Wie viele Test-Queries brauche ich mindestens?

<details><summary>💡 Zeige Antwort</summary>

**Antwort hängt vom Ziel ab:**

**Development (iterative Verbesserung):**
- **Minimum:** 30 Queries
- **Empfohlen:** 50-100 Queries
- **Warum:** Schnelles Feedback, ausreichend für Trend-Erkennung

**Production Validation:**
- **Minimum:** 100 Queries
- **Empfohlen:** 200-500 Queries
- **Warum:** Statistische Signifikanz, robuste Metrik-Werte

**Research / Benchmark:**
- **Minimum:** 500 Queries
- **Empfohlen:** 1000+ Queries
- **Warum:** Publikations-Standard, vergleichbar mit anderen Systemen

**Start klein, erweitere iterativ:**
```
Phase 1: 30 Queries (Quick Start)
Phase 2: 100 Queries (Robustere Eval)
Phase 3: 500+ Queries (Production-Ready)
```

**Merksatz:** "Lieber 50 gute Queries als 500 schlechte"
</details>

### 2. Anwendungsfrage: Binary oder Graded Relevance?

**Frage:** Ich will mein Retrieval-System evaluieren. Soll ich Binary (0/1) oder Graded (0-3) Relevance nutzen?

<details><summary>💡 Zeige Antwort</summary>

**Entscheidungsbaum:**

```
Welche Metriken willst du messen?
├─ Nur Precision@K, Recall@K, MRR
│   └─ Binary Relevance (0/1) reicht! ✅
│       → Schneller zu annotieren
│       → Klar: relevant oder nicht
│
└─ Auch NDCG@K (Ranking-Qualität)
    └─ Graded Relevance (0-3) nötig! ✅
        → Langsamer zu annotieren
        → Differenziert Qualität
```

**Binary wählen wenn:**
- Quick Start (wenig Zeit)
- Klare Relevanz (entweder beantwortet Query oder nicht)
- Precision@K, Recall, MRR reichen

**Graded wählen wenn:**
- Ranking-Optimierung wichtig
- Qualitätsunterschiede relevant ("gute" vs. "perfekte" Antwort)
- NDCG@K messen willst

**Praktischer Tipp:**
Start mit Binary (schneller), upgrade später zu Graded wenn nötig.
Binary GT kannst du später zu Graded erweitern (0→0, 1→Score 2 oder 3).

**Siehe:** [03-ranking-metrics.md](03-ranking-metrics.md#3-normalized-discounted-cumulative-gain-ndcgk) - NDCG braucht Graded
</details>

### 3. Praktische Frage: LLM-generiert - Wie reviewen?

**Frage:** Ich nutze LLMs um GT zu generieren (wie du). Was muss ich beim Review beachten?

<details><summary>💡 Zeige Antwort</summary>

**Review-Checkliste:**

**1. Queries realistisch?**
```python
# ❌ Synthetisch/unnatürlich:
"Erläutern Sie bitte die technischen Spezifikationen..."

# ✅ Realistisch (wie User fragen):
"Welche Maße hat das Gerät?"
```

**2. Relevance-Scores korrekt?**
```python
# Häufiger LLM-Fehler: Übersieht subtile Relevanz

Query: "Energieverbrauch"
LLM markiert:
- "172 kWh/Jahr" → Score 3 ✅
- "Energieeffizienzklasse A+" → Score 0 ❌ (sollte 2 sein!)

→ Manuell korrigieren!
```

**3. Fehlende relevante Chunks?**
```python
# LLM übersieht manchmal Chunks

→ Stichprobe: Lies selbst Chunks durch
→ Gibt es relevante die LLM nicht markiert hat?
```

**4. False Positives?**
```python
# LLM markiert manchmal irrelevante als relevant

→ Bei jedem Score 3: Double-Check
→ Ist das wirklich "highly relevant"?
```

**Workflow:**
```
1. LLM generiert → Speichere als "draft_gt.json"
2. Review 10% Sample → Finde häufige Fehlertypen
3. Korrigiere systematisch → Speichere als "reviewed_gt.json"
4. Second Pass (optional) → Für kritische Use-Cases
5. Finalize → "ground_truth.json"
```

**Time Investment:**
- LLM Generation: 1-2 Stunden für 100 Queries
- Review: 4-8 Stunden für 100 Queries
- **Total:** 5-10 Stunden (vs. 20-30 Stunden manuell!)

**Merksatz:** "Trust but verify - LLM beschleunigt, Review sichert Qualität"
</details>

## 📊 GT-Qualität: Benchmarks

### Wie gut ist deine Ground Truth?

| Metrik | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| **Anzahl Queries** | 30 | 100 | 500+ |
| **Avg. relevant chunks/query** | 1 | 3-5 | 5-10 |
| **Query-Länge Std** | >1.5 words | >2 words | >3 words |
| **Inter-Annotator Agreement (Kappa)** | >0.4 | >0.6 | >0.8 |
| **Edge Cases Coverage** | 5% | 10% | 20% |
| **Review Time (LLM-assisted)** | - | 5-10h per 100 | <5h per 100 |

### Real-World Benchmarks

**BEIR Benchmark (Research Standard):**
- 58,000+ Queries
- 18 verschiedene Datasets
- Binary + Graded Relevance
- Human-annotated

**Dein Projekt (Development):**
- Start: 30-50 Queries ✅
- Development: 100 Queries ✅
- Production: 200-500 Queries ✅

**Vergleich:**
Du brauchst NICHT BEIR-Level für gutes Retrieval-System!
50-100 gut-annotierte Queries >> 1000 schlecht-annotierte.

## 🚀 Was du jetzt kannst

**Konzeptverständnis:**
- ✓ Du verstehst warum Ground Truth essentiell ist
- ✓ Du weißt wie viele Queries du brauchst (30-100 für Dev, 200+ für Prod)
- ✓ Du unterscheidest Binary vs. Graded Relevance

**Praktische Fähigkeiten:**
- ✓ Du erstellst Test-Query-Sets (diversifiziert, realistisch)
- ✓ Du annotierst relevante Chunks (manuell oder LLM-assisted)
- ✓ Du reviewst LLM-generierte GT (Fehlertypen erkennen)
- ✓ Du bewertest GT-Qualität (Inter-Annotator Agreement, Coverage)

**Workflow-Kompetenz:**
- ✓ Du nutzt LLMs zur Beschleunigung (10x schneller)
- ✓ Du kombinierst LLM + Human Review (Hybrid-Ansatz)
- ✓ Du iterierst: Start klein (30), erweitere später (100, 500)

**Nächster Schritt:**
→ GT erstellt? Nutze sie für Evaluation: [03-ranking-metrics.md](03-ranking-metrics.md)

## 🔗 Weiterführende Themen

**GT nutzen für Evaluation:**
→ [03-ranking-metrics.md](03-ranking-metrics.md) - Precision@K, Recall@K, MRR, NDCG berechnen mit deiner GT

**Retrieval-Methoden die du evaluieren kannst:**
→ [04-advanced/retrieval-methods/](../../../04-advanced/02-retrieval-optimization.md) - Dense, Sparse, Hybrid Retrieval

**Automatisierte Evaluation ohne GT:**
→ [08-advanced-techniques.md](../../03-production/08-advanced-techniques.md) - LLM-as-Judge, RAGAS Framework

**Chunk-Qualität vor GT-Erstellung:**
→ [01-chunk-quality.md](01-chunk-quality.md) - Stelle sicher dass deine Chunks gut sind bevor du annotierst

**Key Papers & Resources:**
1. **BEIR Benchmark** (Thakur et al., 2021) - Standard für Retrieval-Evaluation GT
2. **MS MARCO** (Nguyen et al., 2016) - Large-scale GT mit 1M+ Queries
3. **Cohen's Kappa** (Cohen, 1960) - Inter-Annotator Agreement Metrik
4. **Active Learning for Annotation** (Settles, 2009) - Smart Sampling statt alle Chunks
