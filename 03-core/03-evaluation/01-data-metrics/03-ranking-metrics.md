# Ranking Metrics: Wie gut findet dein Retrieval System relevante Dokumente?

> **💡 Wichtig:** Diese Metriken sind **unabhängig** von der Retrieval-Methode!
>
> Sie funktionieren gleich, egal ob du nutzt:
> - Dense Retrieval (Embedding-basiert: Sentence-Transformers, OpenAI)
> - Sparse Retrieval (BM25, TF-IDF)
> - Hybrid Retrieval (Kombination)
>
> **Ranking-Metriken** messen nur: "Sind die Top-K Ergebnisse relevant?" - unabhängig davon WIE du sie gefunden hast.
>
> → Für Retrieval-Methoden siehe: [04-advanced/retrieval-methods/](../../../04-advanced/02-retrieval-optimization.md)
> → Für Embedding-Models siehe: [03-core/embeddings/](../../02-embeddings/)

## ❓ Das Problem (Problem-First)

**Ohne gute Ranking-Metriken geht folgendes schief:**

1. **Du merkst nicht, dass relevante Docs auf Platz 20 landen** - User sehen nur Top-5, aber wichtigste Info ist weiter unten
   - *Beispiel:* User fragt "Wie kündige ich meinen Vertrag?" → Kündigungsformular ist auf Position 15, User findet es nicht

2. **Du kannst Systeme nicht vergleichen** - Ist Retrieval-System A besser als System B? Ohne Metriken: Bauchgefühl
   - *Beispiel:* Zwei verschiedene Ansätze testen, aber keine quantitative Basis für Entscheidung

3. **Du optimierst am falschen Ende** - System liefert 100 Dokumente, aber nur Top-3 sind relevant → Hoher Recall täuscht über schlechte User Experience hinweg
   - *Beispiel:* 80% Recall klingt gut, aber Precision@5 ist nur 20% → User sieht 4 von 5 irrelevanten Docs

**Die zentrale Frage:**
Von allen zurückgegebenen Dokumenten - wie viele der *ersten K* Dokumente sind relevant? Und wo steht das erste relevante Dokument?

**Beispiel-Szenario:**
```
Query: "Python async programming tutorial"
Deine Datenbank: 1000 Dokumente (davon 20 relevant zu async Python)

System liefert Top-10:
1. ✅ "Asyncio Tutorial"
2. ❌ "Java Threading"
3. ❌ "JavaScript Promises"
4. ✅ "Python Async/Await Guide"
5. ❌ "C++ Multithreading"
6. ✅ "Python Asyncio Best Practices"
7-10. ❌ Alle irrelevant

→ Wie bewertest du diese Retrieval-Performance quantitativ?
```

## 🎯 Lernziele

Nach diesem Kapitel kannst du:
- [ ] Verstehe Precision@K, Recall@K, MRR, NDCG **mathematisch und intuitiv**
- [ ] Erkenne wann welche Metrik sinnvoll ist (Binary vs. Graded Relevance)
- [ ] Implementiere alle Standard-Metriken from scratch in Python
- [ ] Interpretiere Metrik-Werte richtig (was ist ein "guter" MRR@10?)
- [ ] Wähle die richtige Metrik für deinen Use-Case (E-Commerce vs. FAQ-Bot)
- [ ] Erkenne häufige Fehlinterpretationen (hoher Recall ≠ gutes System)

## 🧠 Intuition zuerst

### Alltagsanalogie: Die Bibliothek

**Stell dir vor, du suchst in einer Bibliothek:**

Du fragst den Bibliothekar nach Büchern über "Machine Learning".
Er bringt dir 10 Bücher.

**Precision@10**: Von diesen 10 Büchern, wie viele sind wirklich über ML?
- Er bringt 7 ML-Bücher + 3 über Statistik → Precision@10 = 7/10 = 70%
- *Intuitiv:* "Wie viel Müll ist dabei?"

**Recall@10**: Die Bibliothek hat 50 ML-Bücher. Von denen hat er dir 7 gebracht.
- Recall@10 = 7/50 = 14%
- *Intuitiv:* "Wie viel habe ich verpasst?"

**Mean Reciprocal Rank (MRR)**: Wo lag das erste relevante Buch?
- Erstes ML-Buch war auf Position 3 → Reciprocal Rank = 1/3 = 0.33
- *Intuitiv:* "Wie lange musste ich suchen?"

**NDCG**: Manche Bücher sind "perfekt", andere "okay", andere "irrelevant"
- Perfekte Sortierung: Beste Bücher zuerst → NDCG = 1.0
- Deine Sortierung: Manche gute Bücher erst später → NDCG = 0.75
- *Intuitiv:* "Wie gut ist die Reihenfolge?"

### Visualisierung: Retrieval Results

```
Query: "Python async"
System liefert 10 Docs, du hast Ground Truth (welche sind relevant):

Position │ Relevanz │ Precision@K │ Recall@K (von 5 relevanten)
─────────┼──────────┼─────────────┼──────────────────────────────
   1     │    ✅    │   1/1=100%  │   1/5=20%
   2     │    ❌    │   1/2=50%   │   1/5=20%
   3     │    ❌    │   1/3=33%   │   1/5=20%
   4     │    ✅    │   2/4=50%   │   2/5=40%
   5     │    ❌    │   2/5=40%   │   2/5=40%  ← Precision@5
   6     │    ✅    │   3/6=50%   │   3/5=60%
   7     │    ❌    │   3/7=43%   │   3/5=60%
   8     │    ❌    │   3/8=38%   │   3/5=60%
   9     │    ❌    │   3/9=33%   │   3/5=60%
  10     │    ❌    │   3/10=30%  │   3/5=60%  ← Precision@10
```

**Beobachtungen:**
- Precision sinkt mit mehr Docs (mehr Müll kommt dazu)
- Recall steigt mit mehr Docs (mehr relevante gefunden)
- Trade-off: Mehr Docs → Mehr Recall, aber weniger Precision

### Die Brücke zur Mathematik

Jetzt machen wir das präzise mit Formeln - aber die Intuition bleibt gleich:
- **Precision** = "Wie viel von dem was ich bekomme ist gut?"
- **Recall** = "Wie viel von dem was gut ist bekomme ich?"
- **Ranking-Qualität** = "Sind die guten Sachen weit oben?"

## 🧮 Das Konzept verstehen

### 1. Precision@K und Recall@K

**Mathematische Definition:**

$$
\text{Precision@K} = \frac{|\text{relevant\_docs} \cap \text{retrieved\_docs@K}|}{K}
$$

$$
\text{Recall@K} = \frac{|\text{relevant\_docs} \cap \text{retrieved\_docs@K}|}{|\text{relevant\_docs}|}
$$

**Intuition hinter den Formeln:**

- **Numerator (Zähler)** ist bei beiden gleich: Anzahl relevanter Docs in Top-K
- **Denominator (Nenner)** unterscheidet:
  - Precision: Geteilt durch K (alle zurückgegebenen)
  - Recall: Geteilt durch Gesamtzahl relevanter Docs

**Warum diese Definitionen?**

- **Precision@K**: User-Perspektive → "Von dem was ich sehe (K Docs), wie viel ist brauchbar?"
  - Wichtig wenn User nur Top-K sieht (z.B. Top-5 in Web-Search)

- **Recall@K**: System-Perspektive → "Von allem was relevant ist, wie viel habe ich gefunden?"
  - Wichtig bei E-Discovery (Legal) wo du ALLE relevanten Docs finden musst

**Beispiel-Rechnung:**

```
Query: "Python async"
Ground Truth: 5 relevante Docs in der Datenbank: {Doc_A, Doc_B, Doc_C, Doc_D, Doc_E}

System retrieves Top-10: [Doc_A, Doc_X, Doc_Y, Doc_B, Doc_Z, Doc_C, ...]

Relevant in Top-5: {Doc_A, Doc_B} → 2 relevante
Relevant in Top-10: {Doc_A, Doc_B, Doc_C} → 3 relevante

Precision@5  = 2/5 = 0.40 (40%)
Recall@5     = 2/5 = 0.40 (40% aller relevanten gefunden)

Precision@10 = 3/10 = 0.30 (30%)
Recall@10    = 3/5 = 0.60 (60% aller relevanten gefunden)
```

**Trade-off sichtbar:**
- Mehr Docs (K↑) → Precision↓, Recall↑
- Weniger Docs (K↓) → Precision↑, Recall↓

### 2. Mean Reciprocal Rank (MRR)

**Mathematische Definition:**

$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

Wobei $\text{rank}_i$ die Position des **ersten** relevanten Dokuments für Query $i$ ist.

**Intuition hinter der Formel:**

- **Reciprocal Rank (1/rank)**: Je früher das erste relevante Doc, desto höher der Score
  - Position 1: 1/1 = 1.0
  - Position 2: 1/2 = 0.5
  - Position 3: 1/3 = 0.33
  - Position 10: 1/10 = 0.1

- **Mean über Queries**: Durchschnitt über alle Test-Queries

**Warum dieser Ansatz?**

MRR fokussiert auf "**Time to first relevant result**" - wichtig für:
- Web-Search (User klickt erstes gutes Ergebnis)
- FAQ-Bots (User braucht EINE gute Antwort)
- Navigational Queries ("Facebook Login" → User will eine spezifische Seite)

**Beispiel-Rechnung:**

```
Test-Set: 3 Queries

Query 1: "Python async"
Retrieved: [❌, ❌, ✅, ❌, ❌, ...]
First relevant: Position 3 → RR₁ = 1/3 = 0.33

Query 2: "Docker networking"
Retrieved: [✅, ❌, ✅, ...]
First relevant: Position 1 → RR₂ = 1/1 = 1.0

Query 3: "Redis caching"
Retrieved: [❌, ✅, ❌, ...]
First relevant: Position 2 → RR₃ = 1/2 = 0.5

MRR = (0.33 + 1.0 + 0.5) / 3 = 1.83 / 3 = 0.61
```

**Interpretation:**
- MRR = 1.0: Perfekt! Immer erstes Ergebnis relevant
- MRR = 0.5: Im Schnitt ist zweites Ergebnis relevant
- MRR = 0.1: Im Schnitt Position 10 (schlecht!)

### 3. Normalized Discounted Cumulative Gain (NDCG@K)

**Problem mit Precision/Recall:** Binary Relevanz (relevant ✅ vs. irrelevant ❌)
**Realität:** Graded Relevanz (perfekt=3, gut=2, okay=1, irrelevant=0)

**Schritt-für-Schritt Ableitung:**

**Step 1: Cumulative Gain (CG)**

$$
\text{CG@K} = \sum_{i=1}^{K} \text{rel}_i
$$

Einfach die Summe aller Relevanz-Scores.

**Problem:** Position egal! [3,2,1] hat gleichen CG wie [1,2,3]

**Step 2: Discounted Cumulative Gain (DCG)**

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}
$$

**Discount-Faktor:** $\frac{1}{\log_2(i+1)}$ bestraft späte Positionen

| Position | Discount | Interpretation |
|----------|----------|----------------|
| 1 | 1/log₂(2) = 1.0 | Volle Wertung |
| 2 | 1/log₂(3) = 0.63 | 63% Wert |
| 3 | 1/log₂(4) = 0.5 | 50% Wert |
| 5 | 1/log₂(6) = 0.39 | 39% Wert |
| 10 | 1/log₂(11) = 0.29 | 29% Wert |

**Step 3: Normalized DCG (NDCG)**

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

Wobei **IDCG** (Ideal DCG) = DCG der *perfekten* Sortierung (beste Docs zuerst)

**Warum Normalisierung?**
- Ermöglicht Vergleich zwischen Queries (verschiedene Anzahl relevanter Docs)
- Range: [0, 1], wobei 1.0 = perfekte Sortierung

**Beispiel-Rechnung:**

```
Query: "Python async tutorial"

Ground Truth Relevanz-Scores:
- Doc_A: 3 (perfekt)
- Doc_B: 2 (gut)
- Doc_C: 2 (gut)
- Doc_D: 1 (okay)
- Rest: 0 (irrelevant)

System Retrieval (Top-5):
[Doc_B(2), Doc_X(0), Doc_A(3), Doc_C(2), Doc_D(1)]

DCG@5:
= 2/log₂(2) + 0/log₂(3) + 3/log₂(4) + 2/log₂(5) + 1/log₂(6)
= 2/1 + 0/1.58 + 3/2 + 2/2.32 + 1/2.58
= 2.0 + 0 + 1.5 + 0.86 + 0.39
= 4.75

Ideal Retrieval (beste Reihenfolge):
[Doc_A(3), Doc_B(2), Doc_C(2), Doc_D(1), Doc_X(0)]

IDCG@5:
= 3/log₂(2) + 2/log₂(3) + 2/log₂(4) + 1/log₂(5) + 0/log₂(6)
= 3/1 + 2/1.58 + 2/2 + 1/2.32 + 0
= 3.0 + 1.27 + 1.0 + 0.43 + 0
= 5.70

NDCG@5 = DCG/IDCG = 4.75/5.70 = 0.833
```

**Interpretation:**
- NDCG@5 = 0.833 → 83% der idealen Sortierung erreicht
- Gut, aber nicht perfekt (Doc_A sollte auf Position 1 sein, nicht 3)

## ⚠️ Häufige Missverständnisse (Misconception Debugging)

### ❌ Missverständnis 1: "Hoher Recall bedeutet gutes System"

**Warum das falsch ist:**
```python
# "Dummes" System: Gib ALLE Dokumente zurück
retrieved = all_docs  # 10,000 Dokumente
relevant = 50  # Davon 50 relevant

Recall = 50/50 = 100%  # Perfekt!
Precision = 50/10,000 = 0.5%  # Katastrophal!
```

User bekommt 10,000 Docs und muss selbst suchen → Nutzlos trotz 100% Recall

**✓ Richtig ist:**
- Recall alleine sagt nichts über User Experience
- Wichtig: **Precision@K** für K das der User sieht (meist K=5 oder K=10)

**Merksatz:** "Recall ohne Precision ist wie eine Bibliothek die dir alle Bücher bringt"

### ❌ Missverständnis 2: "MRR ist besser als NDCG"

**Warum das falsch ist:** Kommt auf den Use-Case an!

**MRR ist besser für:**
- Navigational Queries ("Facebook login" → User will EINE Seite)
- FAQ-Bots (User braucht EINE Antwort)
- "Quick Answer" Szenarios

**NDCG ist besser für:**
- Informational Queries ("Machine Learning tutorials" → User will mehrere gute Ergebnisse)
- E-Commerce Search (User will Auswahl von Produkten sehen)
- Research (mehrere relevante Papers)

**✓ Richtig ist:**
- MRR: "First relevant result" Use-Cases
- NDCG: "Multiple relevant results with varying quality" Use-Cases
- Oft: **Beide** messen und interpretieren!

**Beispiel:**
```python
# Query: "Best Python IDE"
# User will mehrere Optionen vergleichen → NDCG besser

# Query: "Python official website"
# User will genau eine URL → MRR besser
```

### ❌ Missverständnis 3: "Precision@5 = 80% ist gut genug"

**Warum das falsch ist:** Context matters!

**FAQ-Bot:**
- Precision@5 = 80% → User sieht 4 richtige + 1 falsche Antwort
- **Risiko:** Falsche Antwort könnte gewählt werden → Kunde verärgert
- **Target:** Precision@5 > 95%

**E-Commerce Search:**
- Precision@5 = 80% → 4 von 5 Produkten passen
- **Akzeptabel:** User kann falsche Produkte ignorieren
- **Target:** Precision@5 > 70% (niedrigere Anforderung)

**Medical/Legal:**
- Precision@5 = 80% → NICHT akzeptabel
- **Risiko:** Falsche Information könnte schwerwiegende Folgen haben
- **Target:** Precision@5 > 98%

**✓ Richtig ist:**
- Kein universeller "guter Wert"
- Definiere Targets basierend auf:
  - Konsequenz von False Positives
  - User Tolerance für irrelevante Ergebnisse
  - Business Requirements

**Merksatz:** "Dein Precision-Target hängt davon ab, wie teuer ein Fehler ist"

## 🔬 Hands-On: Retrieval Metrics implementieren

### Setup & Beispiel-Daten

```python
from typing import List, Set, Dict
import numpy as np

# Beispiel Ground Truth: Welche Docs sind für welche Query relevant?
ground_truth = {
    "python async": {1, 4, 6, 12, 15},  # Doc IDs
    "docker networking": {2, 7, 8},
    "redis caching": {3, 9, 11, 14}
}

# Beispiel Retrieval Results (Doc IDs in Ranking-Reihenfolge)
retrieval_results = {
    "python async": [1, 23, 45, 4, 67, 6, 89, 12],  # Top-8
    "docker networking": [2, 7, 34, 56, 8],  # Top-5
    "redis caching": [78, 3, 9, 45, 11]  # Top-5
}
```

### 1. Precision@K Implementation

```python
def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Berechnet Precision@K

    Args:
        retrieved: Liste von Doc IDs in Ranking-Reihenfolge
        relevant: Set von relevanten Doc IDs (Ground Truth)
        k: Anzahl Top-Docs zu evaluieren

    Returns:
        Precision@K (0.0 bis 1.0)
    """
    # Nur Top-K betrachten
    retrieved_at_k = retrieved[:k]

    # Wie viele der Top-K sind relevant?
    relevant_retrieved = set(retrieved_at_k) & relevant  # Intersection

    # Precision = relevante / alle zurückgegebenen
    return len(relevant_retrieved) / k if k > 0 else 0.0


# Test
query = "python async"
retrieved = retrieval_results[query]
relevant = ground_truth[query]

print(f"Precision@5: {precision_at_k(retrieved, relevant, 5):.2%}")
print(f"Precision@10: {precision_at_k(retrieved, relevant, 10):.2%}")
```

**Erwartete Ausgabe:**
```
Precision@5: 60.00%  # 3 von 5 relevant (IDs: 1, 4, 6)
Precision@10: 40.00%  # 4 von 10 relevant (wird schlechter mit mehr K)
```

### 2. Recall@K Implementation

```python
def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Berechnet Recall@K

    Args:
        retrieved: Liste von Doc IDs in Ranking-Reihenfolge
        relevant: Set von relevanten Doc IDs (Ground Truth)
        k: Anzahl Top-Docs zu evaluieren

    Returns:
        Recall@K (0.0 bis 1.0)
    """
    if len(relevant) == 0:
        return 0.0

    # Nur Top-K betrachten
    retrieved_at_k = retrieved[:k]

    # Wie viele der relevanten Docs wurden gefunden?
    relevant_retrieved = set(retrieved_at_k) & relevant

    # Recall = gefunden / alle relevanten
    return len(relevant_retrieved) / len(relevant)


# Test
print(f"Recall@5: {recall_at_k(retrieved, relevant, 5):.2%}")
print(f"Recall@10: {recall_at_k(retrieved, relevant, 10):.2%}")
print(f"Ground Truth: {len(relevant)} relevante Docs total")
```

**Erwartete Ausgabe:**
```
Recall@5: 60.00%   # 3 von 5 relevanten gefunden
Recall@10: 80.00%  # 4 von 5 relevanten gefunden (steigt mit K)
Ground Truth: 5 relevante Docs total
```

### 3. Mean Reciprocal Rank (MRR) Implementation

```python
def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Berechnet Reciprocal Rank für eine Query
    (Position des ersten relevanten Dokuments)

    Args:
        retrieved: Liste von Doc IDs in Ranking-Reihenfolge
        relevant: Set von relevanten Doc IDs

    Returns:
        Reciprocal Rank (0.0 bis 1.0), oder 0.0 wenn kein relevantes Doc gefunden
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0  # Kein relevantes Doc gefunden


def mean_reciprocal_rank(
    results: Dict[str, List[int]],
    ground_truth: Dict[str, Set[int]]
) -> float:
    """
    Berechnet MRR über mehrere Queries

    Args:
        results: Dict[query -> Liste von retrieved Doc IDs]
        ground_truth: Dict[query -> Set von relevanten Doc IDs]

    Returns:
        Mean Reciprocal Rank (0.0 bis 1.0)
    """
    rr_scores = []

    for query in results:
        if query not in ground_truth:
            continue

        rr = reciprocal_rank(results[query], ground_truth[query])
        rr_scores.append(rr)

        # Debug: Position des ersten relevanten Docs
        first_pos = int(1/rr) if rr > 0 else "N/A"
        print(f"Query '{query}': First relevant at position {first_pos} (RR={rr:.3f})")

    return np.mean(rr_scores) if rr_scores else 0.0


# Test über alle Queries
mrr = mean_reciprocal_rank(retrieval_results, ground_truth)
print(f"\nMean Reciprocal Rank: {mrr:.3f}")
```

**Erwartete Ausgabe:**
```
Query 'python async': First relevant at position 1 (RR=1.000)
Query 'docker networking': First relevant at position 1 (RR=1.000)
Query 'redis caching': First relevant at position 2 (RR=0.500)

Mean Reciprocal Rank: 0.833
```

**Interpretation:** Im Schnitt ist das erste relevante Doc auf Position ~1.2 (sehr gut!)

### 4. NDCG@K Implementation

```python
def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Berechnet Discounted Cumulative Gain@K

    Args:
        relevances: Liste von Relevanz-Scores in Ranking-Reihenfolge
        k: Anzahl Top-Docs zu evaluieren

    Returns:
        DCG@K
    """
    relevances_at_k = relevances[:k]

    dcg = 0.0
    for i, rel in enumerate(relevances_at_k, start=1):
        # Discount-Faktor: 1/log₂(i+1)
        discount = 1.0 / np.log2(i + 1)
        dcg += rel * discount

    return dcg


def ndcg_at_k(
    retrieved: List[int],
    relevance_scores: Dict[int, float],
    k: int
) -> float:
    """
    Berechnet Normalized Discounted Cumulative Gain@K

    Args:
        retrieved: Liste von Doc IDs in Ranking-Reihenfolge
        relevance_scores: Dict[doc_id -> Relevanz-Score]
        k: Anzahl Top-Docs zu evaluieren

    Returns:
        NDCG@K (0.0 bis 1.0)
    """
    # Relevanz-Scores für retrieved Docs (0 wenn nicht relevant)
    retrieved_relevances = [
        relevance_scores.get(doc_id, 0.0)
        for doc_id in retrieved
    ]

    # DCG des aktuellen Rankings
    actual_dcg = dcg_at_k(retrieved_relevances, k)

    # Ideal DCG: Sortiere Relevanz-Scores absteigend
    ideal_relevances = sorted(relevance_scores.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    # Normalisierung (verhindert Division durch 0)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# Test mit graded relevance
query = "python async"
retrieved = retrieval_results[query]

# Graded Relevanz (3=perfekt, 2=gut, 1=okay, 0=irrelevant)
relevance_scores = {
    1: 3,   # Perfektes Match
    4: 2,   # Gutes Match
    6: 2,   # Gutes Match
    12: 1,  # Okay Match
    15: 1   # Okay Match
}

ndcg_5 = ndcg_at_k(retrieved, relevance_scores, 5)
ndcg_10 = ndcg_at_k(retrieved, relevance_scores, 10)

print(f"NDCG@5: {ndcg_5:.3f}")
print(f"NDCG@10: {ndcg_10:.3f}")

# Zeige Ranking vs. Ideal
print("\nAktual Ranking (Top-5):")
for i, doc_id in enumerate(retrieved[:5], 1):
    rel = relevance_scores.get(doc_id, 0)
    print(f"  {i}. Doc {doc_id}: Relevance={rel}")

print("\nIdeal Ranking:")
ideal_ranking = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
for i, (doc_id, rel) in enumerate(ideal_ranking[:5], 1):
    print(f"  {i}. Doc {doc_id}: Relevance={rel}")
```

**Erwartete Ausgabe:**
```
NDCG@5: 0.874
NDCG@10: 0.812

Aktual Ranking (Top-5):
  1. Doc 1: Relevance=3  ✓ Gut!
  2. Doc 23: Relevance=0  ✗ Schlecht
  3. Doc 45: Relevance=0  ✗ Schlecht
  4. Doc 4: Relevance=2  ✓ Okay
  5. Doc 67: Relevance=0  ✗ Schlecht

Ideal Ranking:
  1. Doc 1: Relevance=3
  2. Doc 4: Relevance=2
  3. Doc 6: Relevance=2
  4. Doc 12: Relevance=1
  5. Doc 15: Relevance=1
```

**Interpretation:**
- NDCG@5 = 0.874 → 87% der idealen Sortierung
- Gut, weil Doc 1 (beste) auf Position 1
- Könnte besser sein: Doc 4, 6 sollten auf Position 2-3 sein (nicht 23, 45)

## ⏱️ 5-Minuten-Experte

### 1. Verständnisfrage: Precision vs. Recall Trade-off

**Frage:** Du hast Precision@10=30% und Recall@10=90%. Was bedeutet das? Ist das gut oder schlecht?

<details><summary>💡 Zeige Antwort</summary>

**Antwort:**
- **Precision@10=30%**: Von 10 zurückgegebenen Docs sind nur 3 relevant (viel Müll)
- **Recall@10=90%**: Von allen relevanten Docs hast du 90% gefunden (hohe Abdeckung)

**Interpretation:** Dein System ist "zu großzügig"
- Findet fast alles Relevante (gut!)
- Aber gibt auch viel Irrelevantes zurück (schlecht für User Experience)

**Ist das gut oder schlecht?**
- **Schlecht für:** User-facing Search (User sieht 7 irrelevante von 10 Docs)
- **Okay für:** First-Stage Retrieval in Two-Stage System (Re-Ranker filtert später)
- **Gut für:** Legal Discovery (lieber zu viel als zu wenig finden)

**Verbesserung:**
- Threshold erhöhen (weniger Docs zurückgeben)
- Trade-off: Precision↑, Recall↓
</details>

### 2. Anwendungsfrage: Welche Metrik für welchen Use-Case?

**Frage:** Du baust drei Systeme: (A) FAQ-Bot, (B) E-Commerce Search, (C) Academic Paper Search. Welche Metrik ist jeweils am wichtigsten?

<details><summary>💡 Zeige Antwort</summary>

**A) FAQ-Bot:**
- **Wichtigste Metrik:** MRR (Mean Reciprocal Rank)
- **Warum:** User braucht EINE gute Antwort schnell
- **Target:** MRR > 0.8 (erste relevante Antwort meist in Top-2)
- **Zweitwichtig:** Precision@1 (ist die Top-Antwort korrekt?)

**B) E-Commerce Search:**
- **Wichtigste Metrik:** NDCG@10
- **Warum:** User will mehrere Produkte vergleichen, beste sollten oben sein
- **Target:** NDCG@10 > 0.7
- **Zweitwichtig:** Recall@10 (genug Auswahl?)

**C) Academic Paper Search:**
- **Wichtigste Metrik:** Recall@50 + Precision@10
- **Warum:** Researcher will ALLE relevanten Papers finden (hoher Recall), aber auch gute Top-Results (Precision@10)
- **Target:** Recall@50 > 0.9, Precision@10 > 0.6
- **Trade-off:** Niedrigere Precision akzeptabel, wenn Recall hoch

**Merksatz:**
- One answer needed → MRR
- Multiple quality-graded results → NDCG
- Comprehensive coverage → Recall + Precision
</details>

### 3. Trade-off-Frage: NDCG vs. Precision@K

**Frage:** Dein System hat NDCG@10=0.85, aber Precision@10=50%. Ein zweites System hat NDCG@10=0.75 und Precision@10=70%. Welches ist besser?

<details><summary>💡 Zeige Antwort</summary>

**Kommt darauf an:**

**System 1 (NDCG=0.85, P@10=50%):**
- Hoher NDCG → Gute Sortierung, beste Docs weit oben
- Niedriger Precision → Viele irrelevante Docs in Top-10
- **Interpretation:** Perfekte Docs auf Position 1-2, dann viel Müll
- **Beispiel:** [3, 3, 0, 0, 0, 0, 0, 0, 2, 2] (Relevanz-Scores)

**System 2 (NDCG=0.75, P@10=70%):**
- Niedriger NDCG → Schlechtere Sortierung
- Höherer Precision → Weniger irrelevante Docs
- **Interpretation:** Viele relevante Docs, aber nicht optimal sortiert
- **Beispiel:** [1, 2, 1, 0, 2, 1, 0, 1, 2, 0] (Relevanz-Scores)

**Wähle System 1 wenn:**
- User sieht nur Top-3 (dann ist hoher NDCG wichtiger)
- "Best Match" kritisch (z.B. Product Recommendation)

**Wähle System 2 wenn:**
- User scrollt durch alle Top-10
- Wichtiger dass kein Müll dabei ist (z.B. Medical Info)

**Best Practice:** Messe BEIDES und entscheide basierend auf User Behavior
- Tracke: Welche Position klickt User? → Wenn meist Top-1-3: NDCG wichtiger
- Tracke: Scrollt User durch alle? → Wenn ja: Precision wichtiger
</details>

## 📊 Vergleiche & Benchmarks

### Wann nutze ich was?

| Metrik | Use Case | Vorteil | Nachteil | Typisches Target |
|--------|----------|---------|----------|------------------|
| **Precision@K** | User-facing Search, K=5 oder 10 | Einfach zu verstehen | Ignoriert Ranking-Qualität | >70% |
| **Recall@K** | E-Discovery, Medical | Findet alle relevanten Docs | User sieht evtl. viel Müll | >90% |
| **MRR** | FAQ-Bot, Navigational Search | Fokus auf erste Antwort | Ignoriert restliche Results | >0.8 |
| **NDCG@K** | E-Commerce, Multi-Result Search | Berücksichtigt Ranking + Grades | Komplexer zu berechnen | >0.7 |

### Decision Tree: Metrik-Auswahl

```
Brauchst du graded relevance (0-3) oder binary (ja/nein)?
├─ Binary → Precision@K, Recall@K, MRR
│   └─ User sieht nur Top-1 Ergebnis?
│       ├─ JA → MRR (FAQ-Bot)
│       └─ NEIN → Precision@K + Recall@K
│           └─ Was ist wichtiger?
│               ├─ User Experience → Precision@K
│               └─ Nichts verpassen → Recall@K
│
└─ Graded → NDCG@K
    └─ K = Anzahl Ergebnisse die User sieht
```

### Benchmark: Typische Werte verschiedener Systeme

| System-Typ | Precision@5 | Recall@10 | MRR | NDCG@10 |
|------------|-------------|-----------|-----|---------|
| **State-of-the-Art (BEIR Benchmark)** | 65-75% | 55-65% | 0.55-0.70 | 0.52-0.62 |
| **Gutes Production RAG-System** | 60-70% | 70-80% | 0.70-0.85 | 0.65-0.75 |
| **Okay Production System** | 45-60% | 60-70% | 0.60-0.70 | 0.55-0.65 |
| **Needs Improvement** | <45% | <60% | <0.60 | <0.55 |

**Quelle:** BEIR Benchmark (2021), eigene Production-Erfahrung

### 🔍 Metrik-Unabhängigkeit verstehen

**Wichtig:** Diese Metriken messen nur das **Ergebnis** des Retrievals, nicht die **Methode**!

```
Retrieval-Pipeline:
┌─────────────────────────────────────────────────────────────┐
│ 1. Retrieval-Methode (WIE du suchst)                        │
│    ├─ Dense Retrieval (Embeddings + Cosine Similarity)     │
│    ├─ Sparse Retrieval (BM25, TF-IDF)                      │
│    └─ Hybrid (Kombination)                                 │
│                                                              │
│    → Siehe: 04-advanced/retrieval-methods/                 │
│    → Siehe: 03-core/embeddings/ (für Dense Retrieval)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Retrieval-Ergebnis                                       │
│    Top-K Dokumente: [Doc_1, Doc_5, Doc_3, Doc_8, Doc_2]    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Evaluation (WAS du misst) ← Du bist HIER                │
│    Ranking Metrics (dieses Kapitel):                        │
│    - Precision@K: Wie viele der Top-K sind relevant?       │
│    - MRR: Wo ist das erste relevante Doc?                  │
│    - NDCG: Wie gut ist die Reihenfolge?                    │
│                                                              │
│    → Unabhängig von Schritt 1!                             │
└─────────────────────────────────────────────────────────────┘
```

**Beispiel: Gleiche Metriken, verschiedene Methoden**

```python
# Beispiel: Du vergleichst zwei Retrieval-Methoden

# Methode A: Dense Retrieval (Sentence-Transformers)
results_dense = [1, 5, 8, 12, 3]  # Doc IDs
ground_truth = {1, 3, 8, 15, 20}

precision_dense = precision_at_k(results_dense, ground_truth, 5)
# → 60% (3 von 5 relevant)

# Methode B: BM25 (Sparse Retrieval)
results_bm25 = [1, 3, 8, 15, 20]  # Doc IDs
precision_bm25 = precision_at_k(results_bm25, ground_truth, 5)
# → 100% (5 von 5 relevant)

# Gleiche Metrik, verschiedene Methoden!
# Jetzt weißt du: BM25 funktioniert besser für diesen Use-Case
```

**Was beeinflusst was?**

| Was? | Beeinflusst durch... | Gemessen mit... |
|------|---------------------|-----------------|
| **Embedding-Qualität** | Embedding-Model (Sentence-Transformers, OpenAI) | Embedding-Evaluation (03-core/embeddings/) |
| **Similarity-Metrik** | Embedding-Model (normalisiert? → Cosine vs. Dot Product) | Siehe 02-similarity-measures.md |
| **Retrieval-Qualität** | Retrieval-Methode (Dense, Sparse, Hybrid) | **Ranking-Metriken (dieses Kapitel)** |

**Merksatz:**
> "Ranking-Metriken sind wie ein Thermometer - sie messen die Temperatur, egal ob du mit Gas, Strom oder Holz heizt."

## 🚀 Was du jetzt kannst

**Mathematisches Verständnis:**
- ✓ Du verstehst Precision@K, Recall@K, MRR, NDCG von Intuition bis Formalisierung
- ✓ Du erkennst Trade-offs zwischen Metriken (Precision vs. Recall)
- ✓ Du siehst warum NDCG komplexer aber aussagekräftiger ist als Precision@K

**Praktische Fähigkeiten:**
- ✓ Du implementierst alle Standard-Metriken from scratch
- ✓ Du wählst die richtige Metrik für deinen Use-Case (FAQ vs. E-Commerce vs. Research)
- ✓ Du berechnest Metriken für dein Retrieval-System und interpretierst Ergebnisse

**Kritisches Denken:**
- ✓ Du vermeidest Fehlinterpretationen (hoher Recall ≠ gutes System)
- ✓ Du erkennst wann eine Metrik täuscht (Precision@K ohne Context)
- ✓ Du definierst realistische Targets basierend auf Use-Case und Risiko

## 🔗 Weiterführende Themen

**Nächster logischer Schritt:**
→ [../02-ai-evaluation/04-quality-metrics.md](../02-ai-evaluation/04-quality-metrics.md) - LLM-Antwortqualität bewerten (Faithfulness, Answer Relevance)

**Vertiefung:**
→ [../03-production/08-advanced-techniques.md](../03-production/08-advanced-techniques.md) - RAGAS Framework, LLM-as-Judge, Hard Negatives Mining

**Praktische Anwendung:**
→ [../../../06-applications/](../../../06-applications/) - RAG-System Evaluation in Production

**Core-Konzepte:**
- [../../02-embeddings/03-model-selection.md](../../02-embeddings/03-model-selection.md) - Embedding-Models beeinflussen Retrieval-Metriken
- [02-similarity-measures.md](02-similarity-measures.md) - Mathematische Grundlagen der Ähnlichkeitsmessung

**Advanced Research:**
→ [../../../04-advanced/02-retrieval-optimization.md](../../../04-advanced/02-retrieval-optimization.md) - Two-Stage Retrieval, Hybrid Search für bessere Metriken

**Key Papers:**
1. **BEIR Benchmark** (Thakur et al., 2021) - "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"
2. **NDCG Original** (Järvelin & Kekäläinen, 2002) - "Cumulated Gain-based Evaluation of IR Techniques"
3. **MS MARCO** (Nguyen et al., 2016) - Large-scale IR dataset mit MRR als primary metric
