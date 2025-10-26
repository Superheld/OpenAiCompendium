# Embedding / Dense Vector / Embedding Vector

## Quick Definition

Eine mathematische Repräsentation von Text (oder anderen Daten) als Liste von Zahlen in einem hochdimensionalen Raum, die semantische Bedeutung erhält.

**Kategorie:** Vectors & Embeddings
**Schwierigkeit:** Beginner
**Aliases:** Embedding, Dense Vector, Embedding Vector, Representation, Vector Representation

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

Stell dir vor, du möchtest Wörter auf einem Blatt Papier so anordnen, dass ähnliche Wörter nahe beieinander liegen:
- "Hund" und "Katze" liegen nahe (beide Haustiere)
- "Auto" und "Fahrrad" liegen nahe (beide Fahrzeuge)
- "Hund" und "Auto" liegen weit auseinander (keine Ähnlichkeit)

Ein **Embedding** ist genau das - aber in **384-1024 Dimensionen** statt auf einem 2D-Papier. Jedes Wort (oder Satz) wird zu einem Punkt in diesem hochdimensionalen Raum.

**Beispiel:** Der Satz "Laborkühlschrank" wird zu:
```
[0.23, -0.45, 0.12, ..., 0.67]  # 384 Zahlen
```

### Mathematische Formalisierung

Ein Embedding ist ein Vektor $\mathbf{v} \in \mathbb{R}^n$:

$$\mathbf{v} = [v_1, v_2, v_3, \ldots, v_n]$$

wo:
- $n$ = Dimensionalität (typisch 384, 512, 768, 1024, 1536)
- $v_i \in \mathbb{R}$ = i-te Komponente (reelle Zahl)
- $\mathbf{v}$ repräsentiert semantische Bedeutung

**Embedding-Funktion:**
$$f: \text{Text} \rightarrow \mathbb{R}^n$$

Beispiel:
$$f(\text{"Laborkühlschrank"}) = [0.23, -0.45, 0.12, \ldots, 0.67]$$

**Wichtige Eigenschaft:**

Semantisch ähnliche Texte haben **ähnliche Embeddings** (kleiner Abstand/Winkel im Vektorraum).

$$\text{similarity}(\text{"Hund"}, \text{"Katze"}) > \text{similarity}(\text{"Hund"}, \text{"Auto"})$$

### Why It Matters

**Problem ohne Embeddings:**
Computer können nur exakte String-Matches finden:
- Query: "Kühlschrank für Labor"
- Dokument: "Laborkühlschrank"
- **Match: NEIN** ❌ (unterschiedliche Wörter)

**Mit Embeddings:**
- Query-Embedding: `[0.22, -0.44, 0.11, ...]`
- Dokument-Embedding: `[0.23, -0.45, 0.12, ...]`
- Cosine Similarity: **0.95** ✅ (semantisch identisch!)

**Anwendungen:**
- **Semantic Search**: Finde Dokumente nach Bedeutung, nicht Keywords
- **RAG Systems**: Hole relevante Kontexte für LLM
- **Recommendation**: Finde ähnliche Produkte/Artikel
- **Clustering**: Gruppiere semantisch verwandte Texte

### Common Variations

**1. Text Embeddings** (Sätze/Paragraphen)
```python
model.encode("Der Laborkühlschrank ist defekt")
# → [0.23, -0.45, ..., 0.67] (384 dims)
```

**2. Token Embeddings** (einzelne Tokens in LLMs)
```python
# In GPT/BERT: jedes Token → Embedding
"Laborkühlschrank" → [Token1_emb, Token2_emb, ...]
```

**3. Multimodal Embeddings** (Bild + Text in gleichem Raum)
```python
# CLIP: Bild und Text in gemeinsamen Embedding-Space
image_emb = [0.1, 0.2, ...]
text_emb  = [0.11, 0.19, ...]  # ähnlich!
```

**Dimensionalitäten nach Modell:**

| Modell | Dimensionen | Use-Case |
|--------|-------------|----------|
| `all-MiniLM-L6-v2` | 384 | Schnell, gute Balance |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual (DE/EN) |
| `all-mpnet-base-v2` | 768 | Höhere Qualität |
| `text-embedding-ada-002` (OpenAI) | 1536 | State-of-the-Art (API) |
| `e5-large-v2` | 1024 | Open-Source SOTA |

---

## 💻 Code-Beispiel

```python
from sentence_transformers import SentenceTransformer

# 1. Modell laden
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Text → Embedding
text = "Der Laborkühlschrank ist defekt"
embedding = model.encode(text)

print(f"Dimensionen: {len(embedding)}")  # 384
print(f"Erste 5 Werte: {embedding[:5]}")
# [0.23456, -0.45123, 0.12789, 0.88234, -0.34567]

# 3. Mehrere Texte gleichzeitig
texts = [
    "Laborkühlschrank defekt",
    "Kühlschrank im Labor kaputt",
    "Serverausfall im Rechenzentrum"
]
embeddings = model.encode(texts)  # Shape: (3, 384)

# 4. Ähnlichkeit berechnen (Cosine Similarity)
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Ähnlichkeit (1 vs 2): {sim:.3f}")  # ~0.95 (sehr ähnlich!)

sim_diff = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
print(f"Ähnlichkeit (1 vs 3): {sim_diff:.3f}")  # ~0.30 (unähnlich)
```

**Output:**
```
Dimensionen: 384
Erste 5 Werte: [0.23456 -0.45123  0.12789  0.88234 -0.34567]
Ähnlichkeit (1 vs 2): 0.953
Ähnlichkeit (1 vs 3): 0.298
```

---

## 🔗 Related Terms

### **Voraussetzungen** (was du vorher wissen solltest)
- **Vektor (Mathematik)**: $n$-Tupel reeller Zahlen
- **Dimensionalität**: Anzahl der Komponenten im Vektor

### **Baut darauf auf** (was danach kommt)
- **[Cosine Similarity](02-cosine-similarity.md)**: Wie man Embeddings vergleicht
- **[Vector Normalization](03-vector-normalization.md)**: Embeddings für Vergleich vorbereiten
- **[Dense Retrieval](05-dense-retrieval.md)**: Embeddings für Suche nutzen

### **Verwandt**
- **Token** (siehe `05-llm-training/01-token.md`): In LLMs wird jedes Token eingebettet
- **Vector Database**: Speichert Millionen Embeddings effizient (Pinecone, Weaviate, Qdrant)

---

## 📍 Where This Appears

### **Primary Chapter** (detaillierte Erklärung)
- `03-core/02-embeddings/01-vector-fundamentals.md` - Vollständige mathematische Grundlagen
- `03-core/02-embeddings/02-embedding-architectures.md` - Wie Embeddings trainiert werden (BERT, Sentence-BERT)
- `03-core/02-embeddings/03-model-selection.md` - Welches Embedding-Modell wählen?

### **Usage Examples** (praktische Anwendung)
- `06-applications/01-rag-systems.md` (Sektion 2) - Embeddings in RAG
- `06-applications/02-search-systems.md` (Sektion 2) - Dense Retrieval
- `04-advanced/01-retrieval-methods.md` - Dense vs. Sparse vs. Hybrid

### **Mentioned In** (weitere Vorkommen)
- `02-modern-ai/01-llms/01-model-families.md` (Sektion 1.1) - Token Embeddings in LLMs
- `02-modern-ai/03-multimodal/` - Multimodale Embeddings (CLIP)
- `03-core/03-evaluation/02-ai-evaluation/03-retrieval-metrics.md` - Embedding-basierte Retrieval-Evaluation

---

## ⚠️ Common Misconceptions

### ❌ "Embeddings sind einfach Word2Vec"
**Falsch!** Word2Vec (2013) war ein früher Ansatz. Moderne Embeddings:
- **Contextualisiert**: "Bank" (Finanz) vs. "Bank" (Sitz) haben unterschiedliche Embeddings
- **Satz-Level**: Nicht nur Wörter, sondern ganze Sätze/Paragraphen
- **Task-optimiert**: Trainiert für spezifische Aufgaben (Retrieval, QA, etc.)

**Richtig:** Word2Vec war der Anfang. Moderne Sentence-BERT/E5 sind weit überlegen.

### ❌ "Mehr Dimensionen = immer besser"
**Falsch!** Dimensionen haben Trade-offs:

| Dimensionen | Vorteile | Nachteile |
|-------------|----------|-----------|
| 384 | Schnell, wenig Speicher | Etwas weniger Präzision |
| 768 | Gute Balance | Standard-Choice |
| 1536 | Höchste Qualität | 4× mehr Speicher, langsamer |

**Richtig:** Wähle Dimensionen basierend auf **Latenz-Budget** und **Qualitäts-Anforderungen**.

### ❌ "Embedding = Feature Vector"
**Technisch ähnlich, semantisch unterschiedlich:**
- **Feature Vector** (klassisches ML): Hand-crafted Features (TF-IDF, Bag-of-Words)
- **Embedding** (Deep Learning): **Gelernte** Repräsentation, erfasst semantische Bedeutung

**Richtig:** Embeddings sind semantisch reichhaltiger als klassische Features.

---

## 🎯 Zusammenfassung

**Ein Satz:** Embeddings sind Vektoren, die semantische Bedeutung von Text in Zahlen codieren und Maschinen ermöglichen, "Ähnlichkeit" zu verstehen.

**Merksatz:** "Ähnliche Bedeutung → ähnliche Vektoren → kleine Distanz im Embedding-Space"

**Key Takeaway:** Ohne Embeddings keine moderne Semantic Search, kein RAG, keine multimodale AI. Embeddings sind das **Fundament aller retrieval-basierten AI-Systeme**.

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ➡️ [Nächster Begriff: Cosine Similarity](02-cosine-similarity.md)
