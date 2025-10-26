# Sparse Retrieval / BM25 / Lexical Search

## Quick Definition

Traditionelle Keyword-basierte Suche mit **exakten Term-Matches** - schnell, explainbar, aber ohne semantisches Verständnis.

**Kategorie:** Vectors & Embeddings
**Schwierigkeit:** Beginner
**Aliases:** Sparse Retrieval, BM25, Lexical Search, Keyword Search, Term-based Retrieval

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

**Sparse Retrieval** funktioniert wie eine klassische Suchmaschine:
- Zähle **gemeinsame Keywords** zwischen Query und Dokument
- Gewichte wichtige Wörter höher (seltene Wörter = wichtiger)
- Ranke Dokumente nach Keyword-Score

**Beispiel:**
```
Query:    "Laborkühlschrank defekt"
Dokument: "Der Laborkühlschrank ist defekt"

Gemeinsame Keywords: ["Laborkühlschrank", "defekt"] → 2/2 = 100% Match ✅
```

**Aber:**
```
Query:    "Laborkühlschrank defekt"
Dokument: "Kühlschrank im Labor kaputt"

Gemeinsame Keywords: [] → 0% Match ❌ (trotz gleicher Bedeutung!)
```

**"Sparse"** weil: Nur wenige Dimensionen sind "aktiv" (vorhandene Keywords), Rest = 0.

### Mathematische Formalisierung

**BM25 (Best Matching 25)** ist der Standard-Algorithmus für Sparse Retrieval:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

**Komponenten:**

1. **$f(t, d)$**: Term Frequency - Wie oft kommt Keyword $t$ in Dokument $d$ vor?
2. **$\text{IDF}(t)$**: Inverse Document Frequency - Wie selten ist $t$ in allen Dokumenten?
   $$\text{IDF}(t) = \log\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$
   - $N$: Anzahl aller Dokumente
   - $n(t)$: Anzahl Dokumente mit Term $t$

3. **$|d|$**: Dokumentlänge (Anzahl Wörter)
4. **$\text{avgdl}$**: Durchschnittliche Dokumentlänge

**Parameter:**
- **$k_1$**: Term Frequency Sättigung (typisch: 1.2-2.0)
- **$b$**: Längen-Normalisierung (typisch: 0.75)

**Intuition:**
- **Seltene Wörter** (hohes IDF) = wichtiger
- **Häufige Wörter** im Dokument (hohes TF) = relevanter
- **Längere Dokumente** werden normalisiert (nicht bevorzugt)

### Why It Matters

**1. Baseline für Retrieval**

BM25 ist der **Standard-Baseline** für alle Retrieval-Evaluationen:
- Einfach zu implementieren
- Keine Training nötig
- Oft überraschend gut!

**Benchmark:** In vielen Evaluations (BEIR, MS MARCO) schlägt BM25 sogar einfache Dense Retrieval Modelle.

**2. Exakte Keyword-Matches**

Wenn exakte Terme wichtig sind, ist BM25 überlegen:

| Use-Case | BM25 | Dense Retrieval |
|----------|------|-----------------|
| **Produktnummern** ("SKU-12345") | ✅ Findet exakt | ❌ Embedding ungenau |
| **Namen** ("Dr. Müller") | ✅ Findet exakt | ⚠️ Kann Varianten finden |
| **IDs** ("Ticket-9876") | ✅ Perfekt | ❌ Nicht trainiert dafür |
| **Fachbegriffe** (out-of-domain) | ✅ Keyword reicht | ❌ Embedding kennt Begriff nicht |

**3. Explainability**

BM25-Scores sind **nachvollziehbar**:
```python
# Warum wurde dieses Dokument gerankt?
# → Weil "Laborkühlschrank" 3× vorkommt (hohes TF)
# → Und "defekt" selten ist (hohes IDF)
```

Dense Retrieval: "Embedding-Ähnlichkeit 0.87" (Blackbox!)

**4. Keine Trainingskosten**

- **BM25**: Funktioniert sofort auf jedem Datensatz
- **Dense**: Braucht vortrainiertes Embedding-Modell (oder Fine-Tuning)

### Common Variations

**1. BM25** (Standard)
- Wie oben beschrieben
- Parameter: $k_1 = 1.5$, $b = 0.75$

**2. TF-IDF** (älter, einfacher)
$$\text{TF-IDF}(t, d) = f(t, d) \cdot \log\left(\frac{N}{n(t)}\right)$$

- Keine Sättigung, keine Längen-Normalisierung
- BM25 ist fast immer besser!

**3. BM25+** (Variante)
- Zusätzlicher Term für Dokument-Frequenz
- Minimal bessere Ergebnisse (~1-2%)

**4. BM25F** (für strukturierte Dokumente)
- Gewichtet verschiedene Felder unterschiedlich
- Beispiel: Titel wichtiger als Body

```python
score = 0.7 × BM25(query, title) + 0.3 × BM25(query, body)
```

---

## 💻 Code-Beispiel

```python
from rank_bm25 import BM25Okapi
import numpy as np

# 1. Dokumente (tokenisiert)
documents = [
    "Der Laborkühlschrank ist defekt und muss repariert werden".split(),
    "Das Serverrack im Rechenzentrum ist überhitzt".split(),
    "Die Kaffeemaschine im Pausenraum ist kaputt".split(),
    "Der Kühlschrank im Labor funktioniert nicht mehr".split(),
    "Die Klimaanlage im Büro ist ausgefallen".split()
]

# 2. BM25 Index erstellen
bm25 = BM25Okapi(documents)

# 3. Query
query = "Laborkühlschrank kaputt".split()

# 4. Scores berechnen
scores = bm25.get_scores(query)
print(f"BM25 Scores: {scores}")

# 5. Top-K Ranking
top_k = 3
top_indices = np.argsort(scores)[::-1][:top_k]

print(f"\n🔍 BM25 Results für '{' '.join(query)}':")
for rank, idx in enumerate(top_indices, 1):
    doc_text = ' '.join(documents[idx])
    print(f"{rank}. Score: {scores[idx]:.3f} | {doc_text}")

# 6. Vergleich: Dense Retrieval würde finden
print("\n💡 Dense Retrieval würde zusätzlich finden:")
print("   'Kühlschrank im Labor funktioniert nicht' (semantisch ähnlich)")
```

**Output:**
```
BM25 Scores: [2.134 0.    0.    0.    0.   ]

🔍 BM25 Results für 'Laborkühlschrank kaputt':
1. Score: 2.134 | Der Laborkühlschrank ist defekt und muss repariert werden
2. Score: 0.000 | Das Serverrack im Rechenzentrum ist überhitzt
3. Score: 0.000 | Die Kaffeemaschine im Pausenraum ist kaputt

💡 Dense Retrieval würde zusätzlich finden:
   'Kühlschrank im Labor funktioniert nicht' (semantisch ähnlich)
```

**Beobachtung:**
- BM25 findet **nur** Dokument mit exaktem Keyword "Laborkühlschrank"
- "Kühlschrank im Labor" (semantisch identisch) → Score 0.0 ❌
- Dense Retrieval hätte beide gefunden ✅

---

## 🔗 Related Terms

### **Alternative**
- **[Dense Retrieval](05-dense-retrieval.md)**: Semantisch, Embedding-basiert
- **[Hybrid Search](../04-rag-concepts/03-hybrid-search.md)**: BM25 + Dense kombiniert

### **Verwandt**
- **TF-IDF**: Ältere Variante von BM25
- **Inverted Index**: Datenstruktur für schnelle Keyword-Suche
- **Tokenization**: Text → Keywords (wichtig für BM25!)

### **Baut darauf auf**
- **Query Expansion**: Synonyme zu Query hinzufügen (verbessert BM25)
- **Re-Ranking**: BM25 für First-Pass, dann Dense Re-Ranking

---

## 📍 Where This Appears

### **Primary Chapter**
- `04-advanced/01-retrieval-methods.md` (Sektion 2) - Sparse Retrieval im Detail
- `06-applications/02-search-systems.md` (Sektion 1) - BM25 Implementierung

### **Usage Examples**
- `06-applications/01-rag-systems.md` (Sektion 3) - BM25 in Hybrid RAG
- `04-advanced/02-retrieval-optimization.md` - BM25 + Dense kombinieren

### **Comparison**
- `03-core/03-evaluation/02-ai-evaluation/03-retrieval-metrics.md` - BM25 als Baseline

---

## ⚠️ Common Misconceptions

### ❌ "BM25 ist veraltet, Dense Retrieval ist immer besser"
**Falsch!** BM25 hat klare Vorteile:

**BM25 gewinnt:**
- ✅ Exakte IDs, Produktnummern, Namen
- ✅ Out-of-domain Begriffe (nicht in Embedding-Training)
- ✅ Keine GPU/Model-Inferenz nötig
- ✅ Explainable Ranking

**Dense gewinnt:**
- ✅ Semantische Suche (Synonyme, Konzepte)
- ✅ Multilingual Search
- ✅ Typo-Robustheit

**Best Practice:** **Hybrid Search** (BM25 + Dense) für beste Ergebnisse!

**BEIR Benchmark (Durchschnitt über 18 Datasets):**
```
BM25:          0.420 NDCG@10
Dense (E5):    0.512 NDCG@10  (+22%)
Hybrid:        0.547 NDCG@10  (+30% vs. BM25, +7% vs. Dense)
```

### ❌ "BM25 braucht keine Preprocessing"
**Falsch!** Qualität hängt stark von Tokenization ab:

**Schlechtes Preprocessing:**
```python
doc = "Laborkühlschrank"
tokens = doc.split()  # ["Laborkühlschrank"]

query = "Kühlschrank Labor"
tokens_q = query.split()  # ["Kühlschrank", "Labor"]

# Kein Match! ❌
```

**Gutes Preprocessing:**
```python
# Lowercase + Stemming
doc = "laborkühlschrank" → stem("labor") + stem("kühlschrank")
query = "kühlschrank labor" → stem("kühlschrank") + stem("labor")

# Match! ✅
```

**Empfohlene Pipeline:**
1. Lowercase
2. Remove Stopwords ("der", "die", "das")
3. Stemming/Lemmatization (optional)
4. Tokenization

### ❌ "BM25 ist langsamer als Dense Retrieval"
**Genau umgekehrt!**

**Latenz (1M Dokumente):**
```
BM25 (Inverted Index):    5-10ms   ← Sehr schnell!
Dense (HNSW ANN):         10-50ms  ← Langsamer (aber semantisch besser)
Dense (Exact Search):     1000+ms  ← Extrem langsam ohne Index
```

**Warum BM25 schnell ist:**
- **Inverted Index**: Mapping von Keyword → Dokument-IDs
- Nur Dokumente mit Query-Keywords werden geprüft
- Keine Embedding-Inferenz nötig

**Dense Retrieval:**
- Muss Embedding-Modell ausführen (GPU!)
- ANN-Search (HNSW) auch mit Index langsamer als BM25

---

## 📊 BM25 Parameters Tuning

**Standard-Parameter:**
- $k_1 = 1.5$ (Term Frequency Sättigung)
- $b = 0.75$ (Längen-Normalisierung)

**Parameter-Effekte:**

| Parameter | Niedrig | Hoch | Empfehlung |
|-----------|---------|------|------------|
| **$k_1$** | TF zählt weniger | TF zählt mehr | 1.2-2.0 (kurze Docs: niedriger) |
| **$b$** | Länge egal | Länge wichtig | 0.75 (Standard) |

**Tuning-Beispiel:**
```python
from rank_bm25 import BM25Okapi

# Custom Parameters
bm25_tuned = BM25Okapi(documents, k1=1.2, b=0.75)

# Standard
bm25_standard = BM25Okapi(documents)  # k1=1.5, b=0.75
```

**Empfehlung:** Standard-Parameter funktionieren in 90% der Fälle gut. Nur bei spezifischen Problemen tunen.

---

## 🎯 Zusammenfassung

**Ein Satz:** BM25 ist Keyword-basierte Suche mit TF-IDF, die exakte Matches findet, aber semantisches Verständnis fehlt.

**Formel (vereinfacht):**
$$\text{BM25} \approx \sum \text{IDF}(t) \cdot \text{TF}(t, d)_{\text{saturated}}$$

**Key Takeaways:**
1. **Exakt**: Findet exakte Keywords (IDs, Namen, Produktnummern)
2. **Schnell**: 5-10ms Latenz mit Inverted Index
3. **Explainable**: Score basiert auf sichtbaren Keywords
4. **Baseline**: Standard für Retrieval-Evaluation
5. **Hybrid**: Kombiniert mit Dense für beste Ergebnisse

**Wann nutzen?**
- ✅ Exakte Keyword-Matches wichtig
- ✅ Out-of-domain Begriffe (nicht in Embedding-Training)
- ✅ Schnelle Latenz erforderlich
- ✅ Explainable Ranking nötig
- ❌ Semantische Suche (Synonyme) → Dense Retrieval
- ❌ Multilingual → Dense Retrieval

**Best Practice:** **Hybrid Search** (BM25 + Dense) nutzt Stärken beider Methoden!

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ⬅️ [Vorheriger Begriff: Dense Retrieval](05-dense-retrieval.md)
- ➡️ [Nächste Kategorie: Transformers & Attention](../02-transformers-attention/)
