# RAG System: Praktischer Baukasten

## 🎯 Was du danach kannst
- Ein RAG-System für deine eigenen Dokumente entwerfen und bauen
- Die richtige Architektur für verschiedene Use Cases wählen
- Qualität vs. Performance Trade-offs verstehen und optimieren
- Deutsche Texte optimal für RAG aufbereiten

## 📖 Das Problem verstehen

**Situation heute:** Du hast wichtige Dokumente, aber ChatGPT kennt sie nicht. Du musst manuell suchen, kopieren, einfügen - zeitaufwändig und fehleranfällig.

**Was wäre wenn das gelöst wäre?** Eine AI die deine eigenen Dokumente versteht und präzise darauf basierte Antworten gibt. Wie ein persönlicher ChatGPT der deine Firmendokumente, Handbücher oder Forschungsarbeiten "gelesen" hat.

**Konkrete Schmerzpunkte die RAG löst:**
- **Veraltetes Wissen:** ChatGPT kennt nur Training-Daten bis 2023
- **Firmeninterna:** Keine AI kennt deine internen Dokumente
- **Hallucinations:** LLMs erfinden plausible aber falsche "Fakten"
- **Zeitverlust:** Ständiges Copy-Paste zwischen Dokumenten und AI

## 🤔 Chunking-Strategie: Was passt zu deinem Projekt?

### Die Grundfrage
Chunking bestimmt, wie gut dein RAG-System später antwortet. Zu kleine Chunks = präzise aber fragmentierte Antworten. Zu große Chunks = mehr Kontext aber weniger fokussiert.

**Analysiere deine Dokumente erst:**
- **Struktur:** Haben sie klare Abschnitte, Überschriften, Kapitel?
- **Länge:** Kurze Artikel oder lange technische Manuals?
- **Sprache:** Deutsch, Englisch, Fachsprache, gemischt?
- **Zielgruppe:** Brauchen Nutzer exakte Fakten oder allgemeines Verständnis?
- **Update-Häufigkeit:** Statische Dokumente oder häufige Änderungen?

### Entscheidungsbaum
```
Deine Dokumente haben klare Struktur (Überschriften, Abschnitte)?
├─ JA → Semantic Chunking (folgt natürlichen Grenzen)
└─ NEIN → Sind es gleichmäßige Texte (Blogs, News, Artikel)?
    ├─ JA → Fixed-Size mit Overlap (schnell & konsistent)
    └─ NEIN → Recursive Chunking (mehrere Strategien kombiniert)
```

### Strategien im Detail

**Fixed-Size Chunking**
- **Wann verwenden:** Gleichmäßige Texte, schnelle Prototypen, konsistente Performance
- **Vorteile:** Einfach zu implementieren, vorhersagbare Chunk-Größen, schnell
- **Nachteile:** Kann Sätze mittendrin trennen, ignoriert natürliche Strukturen
- **Deutsche Texte:** 400-600 Zeichen optimal (längere deutsche Sätze)
- **Overlap:** 15-25% für Kontext-Erhaltung

```python
# Einfachste Implementierung
chunks = [text[i:i+500] for i in range(0, len(text), 400)]  # 100 Zeichen Overlap
```

**Semantic Chunking**
- **Wann verwenden:** Strukturierte Dokumente, beste Qualität gewünscht, klare Abschnitte
- **Vorteile:** Respektiert natürliche Textgrenzen, bessere Kohärenz, qualitativ hochwertige Chunks
- **Nachteile:** Chunks haben unterschiedliche Größen, komplexere Implementierung
- **Tools:** spaCy für Satz-Grenzen, LangChain RecursiveCharacterTextSplitter

**Recursive Chunking**
- **Wann verwenden:** Gemischte Inhalte, Balance zwischen Qualität und Konsistenz
- **Vorteile:** Probiert mehrere Trennzeichen hierarchisch (Absatz → Satz → Wort)
- **Nachteile:** Komplexer zu konfigurieren, kann unvorhersagbar sein

### Vergleichstabelle für deutsche Texte
| Aspekt | Fixed-Size | Semantic | Recursive |
|--------|------------|----------|-----------|
| Setup-Zeit | ⭐⭐⭐ (10min) | ⭐ (45min) | ⭐⭐ (25min) |
| Chunk-Qualität | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Performance | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Deutsche Texte | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Konsistenz | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| Skalierbarkeit | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

**💡 Deep-Dive:** [chunking-strategies.md](../../03-core/02-embeddings/03-specialization/14-chunking.md)

---

## 🎯 Embedding-Model: Welches passt zu deinem Use Case?

### Die kritische Entscheidung
Das Embedding-Model entscheidet, ob dein RAG-System "BMW X3" und "SUV von BMW" als ähnlich erkennt. Falsche Wahl = schlechte Suchergebnisse trotz perfekter Chunks.

**Analysiere deine Anforderungen:**
- **Sprache:** Nur Deutsch? Multilingual? Fachsprache (Legal, Medical)?
- **Präzision:** Exakte Keywords wichtig oder semantisches Verständnis?
- **Latenz:** Real-time Suche (<200ms) oder Batch-Processing ok?
- **Kosten:** Lokale Models oder API-Calls? Wie viele Queries pro Tag?
- **Domain:** Allgemeine Texte oder spezialisiert (Recht, Medizin, Technik)?

### Entscheidungsbaum für deutsche Texte
```
Brauchst du die absolut beste Qualität?
├─ JA → E5-Large (aber langsamer, größer)
└─ NEIN → Ist Geschwindigkeit kritisch?
    ├─ JA → BGE-Small oder E5-Small
    └─ NEIN → BGE-M3 (guter Kompromiss)

Sind exakte Keywords wichtig?
├─ JA → Dense + SPLADE (Hybrid Approach)
└─ NEIN → Nur Dense Embeddings reichen
```

### Model-Empfehlungen für deutsche RAG-Systeme

**intfloat/multilingual-e5-large**
- **Beste Wahl für:** Höchste Qualität, multilinguale Dokumente
- **Dimensionen:** 1024 (große Vektoren)
- **Geschwindigkeit:** Langsam, aber beste Ergebnisse
- **Deutsch-Performance:** ⭐⭐⭐⭐⭐
- **Use Case:** Qualität wichtiger als Speed

**BAAI/bge-m3**
- **Beste Wahl für:** Guter Kompromiss, gemischte Sprachen
- **Dimensionen:** 1024
- **Geschwindigkeit:** Mittel
- **Deutsch-Performance:** ⭐⭐⭐⭐
- **Besonderheit:** Unterstützt Dense + Sparse + ColBERT

**sentence-transformers/paraphrase-multilingual-mpnet-base-v2**
- **Beste Wahl für:** Bewährte Wahl, stabile Performance
- **Dimensionen:** 768
- **Geschwindigkeit:** Schnell
- **Deutsch-Performance:** ⭐⭐⭐
- **Use Case:** Etablierte Projekte, die Stabilität brauchen

### Code-Snippets
```python
# E5-Large für beste Qualität
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Wichtig: E5 braucht Prefixes
query = "query: Wie funktioniert ein Kühlschrank?"
documents = ["passage: " + doc for doc in documents]

embeddings = model.encode(documents)
```

**💡 Deep-Dive:** [model-selection.md](../../07-infrastructure/14-model-selection.md)

---

## 🔍 Retrieval-Strategie: Wie findest du die besten Chunks?

### Das Herzstück von RAG
Die beste Chunking-Strategie und das perfekte Embedding-Model helfen nichts, wenn dein Retrieval die falschen Dokumente findet. Hier entscheidet sich, ob dein RAG-System präzise antwortet oder irrelevante Informationen liefert.

**Die fundamentale Frage:** Dense Search vs. Sparse Search vs. Hybrid?

### Dense vs. Sparse: Die wichtigste Architektur-Entscheidung

**Dense Search (Embedding-basiert)**
- **Versteht:** Semantische Ähnlichkeit, Synonyme, Paraphrasen
- **Findet:** "Laborkühlschrank" auch wenn Dokument "Medikamentenkühlschrank" enthält
- **Schwäche:** Übersieht exakte Begriffe, Modellnummern, Namen
- **Beispiel:** Query "BMW Geländewagen" findet "SUV von BMW"

**Sparse Search (Keyword-basiert, BM25)**
- **Versteht:** Exakte Wort-Matches, TF-IDF Relevanz
- **Findet:** "HMFvh 4001" exakt, wenn es im Dokument steht
- **Schwäche:** Verpasst Synonyme und semantische Verwandtschaft
- **Beispiel:** Query "BMW Geländewagen" findet NICHT "SUV von BMW"

### Entscheidungsbaum für Retrieval-Strategie
```
Sind exakte Begriffe (Modellnummern, Namen, IDs) kritisch?
├─ JA → Hybrid Search (Dense + Sparse)
└─ NEIN → Sind deine Queries sehr spezifisch?
    ├─ JA → Dense Search reicht
    └─ NEIN → Hybrid für beste Abdeckung

Wie wichtig ist Geschwindigkeit?
├─ KRITISCH → Nur Dense (einfacher)
└─ NICHT KRITISCH → Hybrid (bessere Qualität)
```

### Retrieval-Pattern im Detail

**Naive RAG (Einfach)**
- **Workflow:** Query → Dense Search → Top-K Results → LLM
- **Vorteil:** Schnell zu implementieren, wenig Komplexität
- **Nachteil:** Oft ungenaue Results, keine Query-Optimierung
- **Wann verwenden:** Prototypen, einfache Use Cases

**Advanced RAG (Empfohlen)**
- **Workflow:** Query Expansion → Hybrid Search → Re-Ranking → LLM
- **Query Expansion:** "BMW X3" wird zu "BMW X3, SUV, Geländewagen"
- **Re-Ranking:** Cross-Encoder bewertet Dense+Sparse Results neu
- **Vorteil:** Deutlich bessere Qualität
- **Nachteil:** Komplexer, langsamer

**Agentic RAG (Cutting-Edge)**
- **Workflow:** Query Analysis → Multi-Step Retrieval → Self-Reflection → Answer
- **Self-Reflection:** "Habe ich genug Information? Brauche ich mehr Kontext?"
- **Multi-Step:** Stellt Follow-up Queries basierend auf ersten Results
- **Wann verwenden:** Komplexe Research-Tasks, wenn Qualität absolut kritisch ist

### Code-Beispiele für verschiedene Ansätze
```python
# Naive Dense RAG
results = collection.query(query_embedding, n_results=5)

# Hybrid Search (Dense + Sparse)
dense_results = collection.query(query_embedding, n_results=10)
sparse_results = bm25.get_top_k(query_tokens, k=10)
combined = fusion_score(dense_results, sparse_results, alpha=0.7)

# Query Expansion
expanded_query = query_expander.expand("BMW X3")  # → "BMW X3 SUV Geländewagen"
```

**💡 Deep-Dive:** [search-systems.md](../02-content-understanding/search-systems.md)

---

## 🤖 LLM-Integration: Wie generierst du die finale Antwort?

### Die letzte Meile
Du hast die perfekten Chunks gefunden - aber wie generiert dein LLM daraus eine hilfreiche, präzise Antwort? Hier entscheidet sich User Experience vs. Hallucination Risk.

**Kritische Designentscheidungen:**
- **Welches LLM:** GPT-4, Claude, lokale Models?
- **Prompt Engineering:** Wie strukturierst du Context + Query?
- **Citation:** Wie zeigst du Quellen transparent?
- **Hallucination Prevention:** Wie stellst du sicher, dass nur auf Basis der Chunks geantwortet wird?

### LLM-Auswahl für deutsche RAG-Systeme

**GPT-4 Turbo**
- **Vorteile:** Beste Reasoning, versteht deutschen Kontext sehr gut
- **Nachteile:** Teuer (~$0.01 per 1K tokens), API-abhängig, Datenschutz
- **Wann verwenden:** Höchste Qualität gefordert, Kosten zweitrangig

**Claude 3.5 Sonnet**
- **Vorteile:** Ausgezeichnetes Reasoning, gut für lange Kontexte
- **Nachteile:** API-abhängig, in Europa eingeschränkt verfügbar
- **Besonderheit:** Sehr gut darin, bei mangelnder Information "Ich weiß es nicht" zu sagen

**Lokale Open-Source Models**
- **Llama 3.1 8B:** Gut für deutsche Texte, läuft lokal
- **Mistral 7B:** Schnell, europäischer Anbieter
- **Vorteile:** Datenschutz, keine API-Kosten, vollständige Kontrolle
- **Nachteile:** Niedrigere Qualität, Hardware-Anforderungen

### Prompt-Engineering für RAG

**Strukturiertes Prompt-Template:**
```
Du bist ein hilfsbereiter Assistent. Beantworte die Frage basierend AUSSCHLIESSLICH auf dem bereitgestellten Kontext.

WICHTIG:
- Wenn der Kontext die Antwort nicht enthält, sage "Die Information ist in den bereitgestellten Dokumenten nicht verfügbar"
- Zitiere immer die relevante Quelle mit [Dokument X]
- Erfinde keine zusätzlichen Informationen

KONTEXT:
{retrieved_chunks}

FRAGE: {user_question}

ANTWORT:
```

**Anti-Hallucination Strategien:**
- **Constraint Prompting:** "Antworte NUR basierend auf dem Kontext"
- **Citation Forcing:** "Zitiere IMMER die Quelle für jede Aussage"
- **Confidence Scoring:** "Bewerte deine Antwort von 1-10"

```python
# Einfache LLM-Integration
response = llm.generate(
    prompt=f"Kontext: {chunks}\n\nFrage: {query}\n\nAntwort:",
    max_tokens=300,
    temperature=0.1  # Niedrig für konsistente, faktische Antworten
)
```

**💡 Deep-Dive:** [llm-integration.md](../../03-core/generation/) und [prompt-engineering.md](../../04-advanced/prompt-engineering/)

---

## 📊 Qualität messen: Wie erkennst du ob dein RAG-System gut funktioniert?

### Die kritischen Metriken
Ein RAG-System kann auf viele Arten "schlecht" sein - falsche Chunks gefunden, richtige Chunks aber falsche Antwort, korrekte Antwort aber keine Quellen. Du brauchst systematische Evaluation.

**Die 3 Evaluation-Ebenen:**
1. **Retrieval-Qualität:** Werden die richtigen Chunks gefunden?
2. **Generation-Qualität:** Sind die Antworten korrekt und hilfreich?
3. **End-to-End:** Löst das System das User-Problem?

### Retrieval-Evaluation
```python
# Precision@K: Von den ersten 5 Results, wie viele sind relevant?
precision_5 = relevant_in_top_5 / 5

# Recall: Werden alle relevanten Dokumente gefunden?
recall = relevant_found / total_relevant

# MRR: Position des ersten relevanten Results
mrr = 1 / position_of_first_relevant
```

### Generation-Evaluation mit RAGAS Framework
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Faithfulness: Basiert die Antwort auf den Quellen?
# Answer Relevancy: Beantwortet sie die Frage?
result = evaluate(
    dataset=your_test_data,
    metrics=[faithfulness, answer_relevancy]
)
```

**💡 Deep-Dive:** [evaluation/04-quality-metrics.md](../../03-core/03-evaluation/02-ai-evaluation/04-quality-metrics.md)

### Detailed Pipeline

**1. Document Processing & Indexing**
```python
# Beispiel für deutsche Produktdokumentation
documents = [
    "Der HMFvh 4001 Laborkühlschrank hat 8 Schubfächer mit je 35 Liter Volumen.",
    "Das SafetyDevice aktiviert sich bei Stromausfall und schützt vor Einfrieren.",
    "Die Energieeffizienz beträgt 172 kWh pro Jahr bei Klimaklasse N."
]

# Chunking (siehe CHUNKING-DEEP-DIVE.md)
chunks = chunk_documents(documents, strategy="semantic", max_length=300)

# Embedding (siehe ../05-embeddings/)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large')

# WICHTIG: E5 braucht Prefix!
chunk_texts = ["passage: " + chunk for chunk in chunks]
embeddings = model.encode(chunk_texts, normalize_embeddings=True)

# Vector Database Storage
import chromadb
client = chromadb.Client()
collection = client.create_collection("produktdaten")
collection.add(
    documents=chunks,
    embeddings=embeddings,
    metadatas=[{"source": "produktkatalog", "product_id": extract_id(chunk)} for chunk in chunks],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
```

**2. Query Processing & Retrieval**
```python
def rag_query(user_question: str, top_k: int = 3) -> str:
    # Query Embedding
    query = f"query: {user_question}"  # E5 Prefix
    query_embedding = model.encode(query, normalize_embeddings=True)

    # Similarity Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Context Assembly
    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)

    # LLM Generation
    prompt = f"""
    Beantworte die Frage basierend auf dem gegebenen Kontext.
    Falls der Kontext die Antwort nicht enthält, sage "Information nicht verfügbar".

    Kontext:
    {context}

    Frage: {user_question}

    Antwort:
    """

    # LLM Call (z.B. OpenAI, Anthropic, lokales Model)
    answer = llm.generate(prompt)
    return answer

# Usage
answer = rag_query("Wie viele Schubfächer hat der HMFvh 4001?")
print(answer)  # "Der HMFvh 4001 hat 8 Schubfächer."
```

### RAG vs. Fine-Tuning

| Aspekt | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Wissen Updates** | Einfach (neue Docs hinzufügen) | Schwer (Re-Training nötig) |
| **Kosten** | Niedrig (nur Inference) | Hoch (GPU Training) |
| **Transparenz** | Hoch (Quellen sichtbar) | Niedrig (Black Box) |
| **Latenz** | Höher (Retrieval + Generation) | Niedriger (nur Generation) |
| **Accuracy** | Gut für Fakten | Besser für Style/Format |
| **Use Case** | Knowledge-intensive Tasks | Task-specific Behavior |

**Empfehlung:** RAG für faktische Informationen, Fine-Tuning für Stil und Verhalten.

## 📊 Vergleiche & Varianten

### RAG-Evolutionsstufen

**1. Basic RAG (Naive RAG)**
```
Query → Embed → Search → Retrieve Top-K → LLM → Answer
```
- ✅ Einfach zu implementieren
- ❌ Keine Optimierung für Query oder Context

**2. Advanced RAG**
```
Query → Query Enhancement → Multi-Strategy Retrieval → Re-Ranking → LLM → Answer
```
- ✅ Query Transformation (HyDE, Step-back)
- ✅ Hybrid Search (Dense + Sparse)
- ✅ Re-Ranking mit Cross-Encoder

**3. Modular RAG**
```
Query → Router → Strategy Selection → Specialized Retrieval → Context Optimization → LLM
```
- ✅ Verschiedene Strategien für verschiedene Query-Typen
- ✅ Adaptives Routing basierend auf Query-Analyse

**4. Agentic RAG**
```
Query → Agent Planning → Tool Selection → Multi-Step Retrieval → Synthesis → Answer
```
- ✅ Multi-Hop Reasoning
- ✅ Tool Usage (Calculator, Web Search, APIs)
- ❌ Hohe Komplexität und Latenz

### Performance Comparison

| RAG-Typ | Accuracy | Latenz | Komplexität | Kosten |
|---------|----------|--------|-------------|---------|
| **Basic** | 70% | 1-2s | ⭐ | € |
| **Advanced** | 85% | 3-5s | ⭐⭐⭐ | €€ |
| **Modular** | 90% | 2-4s | ⭐⭐⭐⭐ | €€ |
| **Agentic** | 95% | 10-30s | ⭐⭐⭐⭐⭐ | €€€ |

## 🔬 Implementierung

### Production-Ready RAG System

```python
from dataclasses import dataclass
from typing import List, Optional
import logging

@dataclass
class RAGResult:
    answer: str
    sources: List[str]
    confidence: float
    retrieval_time: float
    generation_time: float

class ProductionRAG:
    def __init__(self, vector_db, embedding_model, llm):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def query(self,
              question: str,
              top_k: int = 5,
              filters: Optional[dict] = None) -> RAGResult:

        import time
        start_time = time.time()

        try:
            # 1. Query Enhancement
            enhanced_query = self.enhance_query(question)

            # 2. Retrieval
            retrieval_start = time.time()
            chunks = self.retrieve(enhanced_query, top_k, filters)
            retrieval_time = time.time() - retrieval_start

            # 3. Context Preparation
            context = self.prepare_context(chunks)

            # 4. Generation
            generation_start = time.time()
            answer = self.generate_answer(question, context)
            generation_time = time.time() - generation_start

            # 5. Post-Processing
            confidence = self.calculate_confidence(answer, chunks)
            sources = [chunk['metadata']['source'] for chunk in chunks]

            # 6. Logging
            self.logger.info(f"RAG query processed: {question[:50]}... "
                           f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")

            return RAGResult(
                answer=answer,
                sources=sources,
                confidence=confidence,
                retrieval_time=retrieval_time,
                generation_time=generation_time
            )

        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            return RAGResult(
                answer="Entschuldigung, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten.",
                sources=[],
                confidence=0.0,
                retrieval_time=0.0,
                generation_time=0.0
            )

    def enhance_query(self, query: str) -> str:
        """Query Enhancement für bessere Retrieval-Ergebnisse"""
        # Beispiel: Query Expansion mit Synonymen
        expansions = {
            "Kühlschrank": ["Kühlgerät", "Kühlaggregat", "Refrigerator"],
            "Schubfächer": ["Schubladen", "Fächer", "Compartments"],
            "Energieverbrauch": ["Stromverbrauch", "kWh", "Energieeffizienz"]
        }

        enhanced = query
        for term, synonyms in expansions.items():
            if term.lower() in query.lower():
                enhanced += f" {' '.join(synonyms)}"

        return enhanced

    def retrieve(self, query: str, top_k: int, filters: Optional[dict]) -> List[dict]:
        """Hybrid Retrieval: Dense + Sparse"""
        # Dense Retrieval
        query_emb = self.embedding_model.encode(f"query: {query}")
        dense_results = self.vector_db.query(
            query_embeddings=[query_emb],
            n_results=top_k * 2,
            where=filters
        )

        # Optional: Sparse Retrieval (BM25) und Fusion
        # sparse_results = self.bm25_search(query, top_k * 2)
        # fused_results = reciprocal_rank_fusion([dense_results, sparse_results])

        return dense_results['documents'][0][:top_k]

    def prepare_context(self, chunks: List[str]) -> str:
        """Context Preparation mit Deduplication"""
        # Entferne ähnliche Chunks (optional)
        unique_chunks = self.deduplicate_chunks(chunks)

        # Formatiere für LLM
        context_parts = []
        for i, chunk in enumerate(unique_chunks, 1):
            context_parts.append(f"[Quelle {i}]\n{chunk}")

        return "\n\n".join(context_parts)

    def generate_answer(self, question: str, context: str) -> str:
        """LLM Generation mit verbessertem Prompt"""
        prompt = f"""Du bist ein hilfsreicher Assistent für Produktinformationen.

Beantworte die folgende Frage basierend ausschließlich auf dem gegebenen Kontext.

Regeln:
- Verwende nur Informationen aus dem Kontext
- Falls die Information nicht verfügbar ist, sage: "Diese Information ist nicht verfügbar"
- Gib Quellenverweise an (z.B. "laut Quelle 1")
- Sei präzise und faktual

Kontext:
{context}

Frage: {question}

Antwort:"""

        return self.llm.generate(prompt, max_tokens=300)

    def calculate_confidence(self, answer: str, chunks: List[str]) -> float:
        """Einfache Confidence-Berechnung"""
        if "nicht verfügbar" in answer.lower():
            return 0.2
        elif len(chunks) >= 3:
            return 0.9
        elif len(chunks) >= 1:
            return 0.7
        else:
            return 0.3

    def deduplicate_chunks(self, chunks: List[str]) -> List[str]:
        """Entferne sehr ähnliche Chunks"""
        # Vereinfachte Implementierung
        unique = []
        for chunk in chunks:
            if not any(self.similarity(chunk, existing) > 0.8 for existing in unique):
                unique.append(chunk)
        return unique

    def similarity(self, text1: str, text2: str) -> float:
        """Einfache Text-Ähnlichkeit"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return len(words1 & words2) / len(words1 | words2)
```

### Monitoring und Evaluation

```python
class RAGMonitoring:
    def __init__(self):
        self.metrics = []

    def log_query(self, query: str, result: RAGResult, user_feedback: Optional[str] = None):
        """Log für spätere Analyse"""
        self.metrics.append({
            'timestamp': time.time(),
            'query': query,
            'answer': result.answer,
            'sources_count': len(result.sources),
            'confidence': result.confidence,
            'retrieval_time': result.retrieval_time,
            'generation_time': result.generation_time,
            'user_feedback': user_feedback  # thumbs_up/thumbs_down
        })

    def get_performance_stats(self) -> dict:
        """Performance-Statistiken"""
        if not self.metrics:
            return {}

        retrieval_times = [m['retrieval_time'] for m in self.metrics]
        generation_times = [m['generation_time'] for m in self.metrics]
        confidences = [m['confidence'] for m in self.metrics]

        return {
            'total_queries': len(self.metrics),
            'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times),
            'avg_generation_time': sum(generation_times) / len(generation_times),
            'avg_confidence': sum(confidences) / len(confidences),
            'satisfaction_rate': len([m for m in self.metrics if m.get('user_feedback') == 'thumbs_up']) /
                               len([m for m in self.metrics if m.get('user_feedback')])
        }
```

### Best Practices für Deutsche RAG-Systeme

**1. Model-Auswahl:**
```python
# Empfohlen für deutsches RAG
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')  # ⭐ Top Choice
# Alternative: BAAI/bge-m3, paraphrase-multilingual-mpnet-base-v2

# LLM Optionen
llm_options = [
    "gpt-4-turbo",              # ⭐ Beste Qualität, teuer
    "claude-3-haiku",          # ⭐ Gute Balance, günstig
    "mistral-medium",          # Open-source Alternative
    "llama-2-70b-chat"        # Selbst-gehostet möglich
]
```

**2. Chunking für deutsche Texte:**
```python
# Optimiert für deutsche Produktdokumentation
def german_chunking_strategy(document: str) -> List[str]:
    import re

    # 1. Nach Absätzen splitten
    paragraphs = document.split('\n\n')

    chunks = []
    for para in paragraphs:
        # 2. Lange Absätze nach Sätzen splitten
        if len(para) > 500:
            sentences = re.split(r'[.!?]+', para)
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk + sentence) < 300:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "

            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks.append(para)

    return [c for c in chunks if len(c) > 50]  # Filter zu kurze Chunks
```

**3. Query Enhancement für deutsche Fachbegriffe:**
```python
def enhance_german_query(query: str) -> str:
    """Erweitere deutsche Queries mit Fachbegriffen und Synonymen"""
    medical_synonyms = {
        "Kühlschrank": ["Kühlgerät", "Kühlaggregat", "Refrigerator", "Kühlschrank"],
        "Medikament": ["Pharmazeutikum", "Arzneimittel", "Medizin", "Drug"],
        "Labor": ["Laboratorium", "Lab", "Laboratory"],
        "Temperatur": ["Grad", "°C", "Temperature", "Temp"],
        "Überwachung": ["Monitoring", "Control", "Kontrolle", "Watch"]
    }

    enhanced = query
    words = query.lower().split()

    for word in words:
        for term, synonyms in medical_synonyms.items():
            if word in term.lower() or term.lower() in word:
                enhanced += f" {' '.join(synonyms)}"

    return enhanced
```

## 🎓 Weiterführende Themen

### Original Papers
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) - Lewis et al. 2020 (Grundpapier)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) - Karpukhin et al. 2020 (Dense Retrieval)
- [FiD: Fusion-in-Decoder](https://arxiv.org/abs/2007.01282) - Izacard & Grave 2020 (Multiple Documents)
- [Self-RAG](https://arxiv.org/abs/2310.11511) - Asai et al. 2023 (Reflective RAG)
- [Corrective RAG](https://arxiv.org/abs/2401.15884) - Yan et al. 2024 (Error Correction)

### Verwandte Kapitel
- **[../05-embeddings/](../05-embeddings/)** - Embedding-Grundlagen für Retrieval
- **[CHUNKING-DEEP-DIVE.md](CHUNKING-DEEP-DIVE.md)** - Optimales Chunking für RAG
- **[RAG-APPROACHES.md](RAG-APPROACHES.md)** - Advanced RAG Patterns
- **[../core/evaluation/08-rag-evaluation.md](../core/evaluation/08-rag-evaluation.md)** - RAG-Evaluation
- **[../core/infrastructure/03-vector-databases.md](../core/infrastructure/03-vector-databases.md)** - Vector Storage

### Nächste Schritte im Lernpfad

**Für Basic RAG Implementation:**
1. [../05-embeddings/02-DENSE-EMBEDDINGS.md](../05-embeddings/02-DENSE-EMBEDDINGS.md) - Model-Auswahl
2. [CHUNKING-DEEP-DIVE.md](CHUNKING-DEEP-DIVE.md) - Chunking-Strategien
3. [../core/infrastructure/03-vector-databases.md](../core/infrastructure/03-vector-databases.md) - Vector Storage

**Für Advanced RAG:**
1. [RAG-APPROACHES.md](RAG-APPROACHES.md) - Advanced Patterns
2. [../core/evaluation/08-rag-evaluation.md](../core/evaluation/08-rag-evaluation.md) - Evaluation
3. [02-agents.md](02-agents.md) - Agentic Capabilities

**Für Production RAG:**
1. [../core/evaluation/](../core/evaluation/) - Comprehensive Evaluation
2. [../core/infrastructure/](../core/infrastructure/) - Production Infrastructure
3. [../ethics/05-safety-guardrails.md](../ethics/05-safety-guardrails.md) - Safety Considerations

## 📚 Ressourcen

### Wissenschaftliche Papers
- [A Survey on RAG](https://arxiv.org/abs/2312.10997) - Guo et al. 2023 (Comprehensive Overview)
- [RAG vs Fine-tuning](https://arxiv.org/abs/2401.08406) - Rogers et al. 2024 (Comparison Study)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) - Liu et al. 2023 (Context Length Effects)

### Blog Posts & Tutorials
- [RAG from Scratch](https://blog.langchain.dev/rag-from-scratch/) - LangChain Comprehensive Guide
- [Advanced RAG Techniques](https://eugeneyan.com/writing/llm-patterns/) - Eugene Yan Pattern Collection
- [Production RAG](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) - Anyscale Guide

### Videos & Talks
- [RAG End-to-End](https://www.youtube.com/watch?v=BrsocJb-fAo) - Complete Implementation Tutorial
- [Advanced RAG Patterns](https://www.youtube.com/watch?v=jz7Z8CmtlWo) - Production Examples
- [RAG vs Fine-tuning Debate](https://www.youtube.com/watch?v=zOFGdqOGUrM) - When to use what

### Frameworks & Tools
- **[LangChain](https://langchain.com/)** - Most comprehensive RAG framework
- **[LlamaIndex](https://llamaindex.ai/)** - RAG-focused, easier to start
- **[Haystack](https://haystack.deepset.ai/)** - Production-ready, modular
- **[RAGAS](https://ragas.readthedocs.io/)** - RAG evaluation framework
- **[ChromaDB](https://www.trychroma.com/)** - Simple vector database for prototyping
- **[Qdrant](https://qdrant.tech/)** - Production vector database

### Implementierungs-Beispiele
- **[RAG Template (LangChain)](https://github.com/langchain-ai/rag-from-scratch)** - Complete Implementation
- **[Production RAG (Zilliz)](https://github.com/zilliztech/GPTCache)** - Caching Layer for RAG
- **[German RAG Example](https://github.com/deepset-ai/haystack-tutorials)** - Deutsch-optimierte Examples

### Community & Support
- **[LangChain Community](https://discord.gg/langchain)** - Discord für Framework-spezifische Fragen
- **[r/MachineLearning](https://reddit.com/r/MachineLearning)** - General ML Discussions
- **[Hugging Face Forums](https://discuss.huggingface.co/)** - Embedding und Model-spezifische Fragen