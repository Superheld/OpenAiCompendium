# 📚 Glossar: Zentrale Begriffsdefinitionen

## 🎯 Ziel

Dieses Glossar ist der **zentrale Referenzpunkt** für alle technischen Begriffe im ML Kompendium. Es löst drei zentrale Probleme:

1. **Konsistenz**: Begriffe wie "Embedding" oder "Quantization" werden einheitlich definiert
2. **Fokus**: Hauptkapitel können sich auf Konzepte konzentrieren, Details stehen hier
3. **Navigation**: Ein Begriff → eine Definition → alle Vorkommen im Kompendium verlinkt

### Was du hier findest

- **28 Kernbegriffe** von "Embedding" bis "Hallucination"
- **Mathematische Notation** konsistent erklärt
- **Code-Beispiele** für praktische Begriffe
- **Cross-References** zu verwandten Konzepten
- **Alias-Tracking** (z.B. "Dense Vector" = "Embedding")

### Was du hier NICHT findest

- **Vollständige Tutorials** → siehe Hauptkapitel
- **Historischen Kontext** → siehe `01-historical/`
- **Produktions-Implementierungen** → siehe `06-applications/`

---

## 📂 Kategorien

Das Glossar ist in **6 thematische Kategorien** organisiert:

### **01. Vectors & Embeddings** [`01-vectors-embeddings/`](01-vectors-embeddings/)
*Grundlagen der Vektorrepräsentation*

**Begriffe (6):**
- **Embedding** / Dense Vector / Embedding Vector
- **Cosine Similarity** / Kosinusähnlichkeit
- **Vector Normalization** / L2-Normalisierung
- **Dot Product** / Inner Product / Skalarprodukt
- **Dense Retrieval** / Neural Search
- **Sparse Retrieval** / BM25 / Lexical Search

**Kernfrage:** Wie werden Texte zu Vektoren und wie vergleicht man sie?

---

### **02. Transformers & Attention** [`02-transformers-attention/`](02-transformers-attention/)
*Transformer-Architektur und Attention-Mechanismen*

**Begriffe (5):**
- **Self-Attention** / Attention Mechanism
- **Multi-Head Attention**
- **Positional Encoding** / RoPE / ALiBi
- **Transformer Block** / Transformer Layer
- **Context Window** / Sequence Length / KV-Cache

**Kernfrage:** Wie funktionieren Transformer und warum sind sie so mächtig?

---

### **03. Quantization & Optimization** [`03-quantization-optimization/`](03-quantization-optimization/)
*Modelloptimierung und Ressourcen-Effizienz*

**Begriffe (7):**
- **Quantization** / Model Quantization
- **Precision Types** (FP32, FP16, BF16, INT8, INT4)
- **GGUF** / llama.cpp Format
- **GPTQ** / GPU-optimized Quantization
- **AWQ** / Activation-aware Weight Quantization
- **LoRA** / Low-Rank Adaptation
- **QLoRA** / Quantized LoRA
- **Mixture of Experts (MoE)**

**Kernfrage:** Wie macht man große Modelle effizienter ohne Qualitätsverlust?

---

### **04. RAG Concepts** [`04-rag-concepts/`](04-rag-concepts/)
*Retrieval-Augmented Generation Patterns*

**Begriffe (5):**
- **RAG** / Retrieval-Augmented Generation
- **Chunking** / Chunk Size / Overlap
- **Hybrid Search** / Hybrid Retrieval
- **Re-Ranking** / Cross-Encoder
- **Relevance vs. Similarity**

**Kernfrage:** Wie baut man RAG-Systeme, die korrekte und relevante Antworten liefern?

---

### **05. LLM Training & Generation** [`05-llm-training/`](05-llm-training/)
*Training, Alignment und Text-Generierung*

**Begriffe (5):**
- **Token** / Tokenization / Vocabulary
- **Next-Token Prediction** / Autoregressive Generation
- **Fine-Tuning** / Instruction Tuning / RLHF
- **Hallucination** / Factual Grounding
- **Prompt Engineering**

**Kernfrage:** Wie werden LLMs trainiert und wie generieren sie Text?

---

### **06. Evaluation Metrics** [`06-evaluation-metrics/`](06-evaluation-metrics/)
*Metriken zur Bewertung von ML/AI-Systemen*

**Begriffe (5):**
- **Precision@K** / Recall@K
- **MRR** / Mean Reciprocal Rank
- **NDCG** / Normalized Discounted Cumulative Gain
- **Perplexity**
- **Faithfulness** / Answer Correctness

**Kernfrage:** Wie misst man die Qualität von Retrieval- und Generierungs-Systemen?

---

## 🔥 Meistgenutzte Begriffe

Diese Begriffe erscheinen in 10+ Kapiteln und sind fundamental für das Verständnis:

| Begriff | Vorkommen | Kategorie | Schwierigkeit |
|---------|-----------|-----------|---------------|
| **Embedding** | 19 Dateien | Vectors | Beginner |
| **Token** | 18 Dateien | LLM Training | Beginner |
| **Quantization** | 11 Dateien | Optimization | Intermediate |
| **Chunking** | 12 Dateien | RAG | Beginner |
| **Self-Attention** | 15 Dateien | Transformers | Intermediate |
| **Cosine Similarity** | 14 Dateien | Vectors | Beginner |
| **Context Window** | 13 Dateien | Transformers | Beginner |

---

## 🗂️ Alias-Index

Viele Begriffe haben mehrere Namen. Hier findest du alle Varianten:

**A**
- **Attention** → siehe Self-Attention
- **Autoregressive** → siehe Next-Token Prediction
- **AWQ** → siehe Quantization Methods

**B**
- **BF16** → siehe Precision Types
- **BM25** → siehe Sparse Retrieval

**C**
- **Cosine Distance** → siehe Cosine Similarity
- **Cross-Encoder** → siehe Re-Ranking

**D**
- **Dense Embedding** → siehe Embedding
- **Dense Vector** → siehe Embedding
- **Dot Product** → siehe Vector Operations

**E**
- **Embedding Space** → siehe Embedding
- **Embedding Vector** → siehe Embedding

**F**
- **Factual Grounding** → siehe Hallucination
- **FP16** / **FP32** → siehe Precision Types
- **Fine-Tuning** → siehe LLM Training

**G**
- **GGUF Format** → siehe Quantization Methods
- **GPTQ** → siehe Quantization Methods

**H**
- **Hybrid Retrieval** → siehe Hybrid Search

**I**
- **Inner Product** → siehe Dot Product
- **INT4** / **INT8** → siehe Precision Types
- **Instruction Tuning** → siehe Fine-Tuning

**K**
- **KV-Cache** → siehe Context Window

**L**
- **L2 Norm** → siehe Vector Normalization
- **Lexical Search** → siehe Sparse Retrieval
- **LoRA** → siehe Model Optimization
- **Low-Rank Adaptation** → siehe LoRA

**M**
- **MoE** → siehe Mixture of Experts
- **Multi-Head Attention** → siehe Transformer Attention

**N**
- **Neural Search** → siehe Dense Retrieval
- **Next-Token Prediction** → siehe LLM Generation
- **Normalization** → siehe Vector Normalization

**P**
- **Positional Encoding** → siehe Transformer Position
- **Precision** → siehe Quantization
- **Prompt** → siehe Prompt Engineering

**Q**
- **Quantization** → siehe Model Quantization
- **QLoRA** → siehe LoRA

**R**
- **RAG** → siehe Retrieval-Augmented Generation
- **Re-Ranking** → siehe Retrieval Optimization
- **Relevance** → siehe Relevance vs. Similarity
- **Retrieval** → siehe Dense/Sparse/Hybrid Retrieval
- **RLHF** → siehe Fine-Tuning
- **RoPE** → siehe Positional Encoding

**S**
- **Scalar Product** → siehe Dot Product
- **Self-Attention** → siehe Attention Mechanism
- **Semantic Search** → siehe Dense Retrieval
- **Sequence Length** → siehe Context Window
- **Similarity** → siehe Cosine Similarity oder Relevance

**T**
- **Token** → siehe Tokenization
- **Transformer** → siehe Transformer Block

**V**
- **Vector** → siehe Embedding
- **Vector Database** → siehe Dense Retrieval (erwähnt in Embeddings)
- **Vocabulary** → siehe Token

---

## 🎓 Wie man das Glossar nutzt

### **Als Anfänger**
1. **Start:** Lies die "Meistgenutzte Begriffe" oben
2. **Reihenfolge:**
   - Embedding → Token → Chunking
   - Dann: Self-Attention → Context Window
3. **Lernpfad:** Folge den "Related Terms" Links innerhalb der Definitionen

### **Als fortgeschrittener Lernender**
- Nutze das Glossar als **Quick Reference** während du Hauptkapitel liest
- Überspringe Basic-Definitionen, fokussiere auf "Why It Matters" und Trade-offs
- Nutze Code-Beispiele zum Experimentieren

### **Als Entwickler**
- **Notation nachschlagen**: Mathematische Formeln konsistent verwenden
- **API-Referenz**: Code-Snippets für schnelle Integration
- **Performance-Hinweise**: Memory/Speed Trade-offs in "Why It Matters"

### **Für Research**
- **Paper-Terminology**: Mapping zwischen Paper-Namen und Kompendium-Begriffen
- **Benchmark-Kontext**: Welche Metriken für welche Evaluation?
- **Cross-References**: Von Begriff zu verwandten Papers/Kapiteln

---

## 📊 Struktur der Glossar-Einträge

Jeder Begriff folgt diesem Template:

### **[Begriff-Name]**

**Quick Definition** (1 Satz, <15 Wörter)
- Kategorie: [Vector/Transformer/RAG/etc.]
- Schwierigkeit: [Beginner/Intermediate/Advanced]
- Aliases: [Alternative Namen]

**Detaillierte Erklärung**
- Intuitive Erklärung (Analogie, ohne Formeln)
- Mathematische Formalisierung (wenn relevant)
- Why It Matters (praktische Bedeutung)
- Common Variations (Varianten des Konzepts)

**Code-Beispiel** (wenn praktisch)
```python
# Minimal, lauffähig, 3-5 Zeilen
```

**Related Terms** (Cross-References)
- Verwandte Begriffe mit Links
- Voraussetzungen (was man vorher wissen sollte)
- Weiterführend (was darauf aufbaut)

**Where This Appears**
- Hauptkapitel (detailed explanation)
- Weitere Vorkommen (references)

---

## 🔗 Integration mit Hauptkapiteln

### **Von Kapitel → Glossar**
Hauptkapitel verlinken auf das Glossar für:
- **Schnelle Definition**: Wenn ein Begriff kurz erwähnt wird
- **Notation-Check**: Mathematische Symbole nachschlagen
- **Prerequisite**: "Für dieses Kapitel solltest du [Begriff] kennen"

### **Von Glossar → Kapitel**
Jeder Glossar-Eintrag verlinkt auf:
- **Primary Chapter**: Wo der Begriff detailliert erklärt wird
- **Usage Examples**: Wo der Begriff in der Praxis vorkommt
- **Advanced Topics**: Weiterführende Kapitel

### **Beispiel-Flow**
```
User liest: 06-applications/01-rag-systems.md
→ Sieht: "Chunking" (mit Glossar-Link)
→ Klickt auf: 08-glossary/04-rag-concepts/02-chunking.md
→ Bekommt: Quick Definition + Code + Link zu 04-advanced/02-retrieval-optimization.md
→ Entscheidet: Quick Definition reicht ODER deep dive ins Advanced-Kapitel
```

---

## 🚀 Was du nach dem Glossar kannst

Nach Durcharbeiten der Glossar-Kategorien kannst du:

### **Vectors & Embeddings**
- ✅ Erklären warum Embeddings semantische Suche ermöglichen
- ✅ Cosine Similarity von Hand berechnen
- ✅ Verstehen wann Dense vs. Sparse Retrieval besser ist

### **Transformers & Attention**
- ✅ Attention-Mechanismus mathematisch nachvollziehen
- ✅ Context Window Limits bei der Architektur-Wahl berücksichtigen
- ✅ Positional Encoding Varianten (RoPE, ALiBi) unterscheiden

### **Quantization & Optimization**
- ✅ GGUF vs. GPTQ vs. AWQ für dein Use-Case wählen
- ✅ Memory-Footprint eines Modells abschätzen
- ✅ LoRA/QLoRA für Finetuning einsetzen

### **RAG Concepts**
- ✅ Chunking-Strategien evaluieren
- ✅ Relevance vs. Similarity Unterschied erklären
- ✅ Hybrid Search mit Re-Ranking implementieren

### **LLM Training**
- ✅ Token-Counting für Cost-Estimation nutzen
- ✅ Hallucination-Risiken einschätzen
- ✅ Prompt Engineering Techniken anwenden

### **Evaluation**
- ✅ Richtige Metriken für Retrieval (NDCG) vs. Generation (Perplexity) wählen
- ✅ Precision@K vs. Recall@K Trade-off verstehen
- ✅ RAG-System mit Faithfulness/Correctness evaluieren

---

## 🔬 Contribution & Updates

Das Glossar ist ein **lebendes Dokument**:

### **Wenn neue Begriffe auftauchen:**
1. Prüfe ob Begriff in 3+ Kapiteln vorkommt
2. Erstelle Glossar-Eintrag nach Template
3. Verlinke von allen Vorkommen

### **Wenn Definitionen sich ändern:**
1. Update nur im Glossar (Single Source of Truth)
2. Alle Kapitel übernehmen automatisch neue Definition

### **Wenn Notation inkonsistent:**
1. Entscheide im Glossar für Standard-Notation
2. Markiere Varianten in "Common Variations"
3. Update Hauptkapitel zur Konsistenz

---

## 📜 Lizenz & Nutzung

Wie das gesamte ML Kompendium:
- ✅ Frei nutzbar für Lernen und Lehre
- ✅ Weitergabe unter gleichen Bedingungen
- ✅ Kommerzielle Nutzung erlaubt
- ⚠️ Mit Quellenangabe

---

**Navigation:**
- 🏠 [Zurück zur Hauptübersicht](../00-overview.md)
- 📖 [Kategorie 1: Vectors & Embeddings](01-vectors-embeddings/)
- 🧠 [Kategorie 2: Transformers & Attention](02-transformers-attention/)
- ⚙️ [Kategorie 3: Quantization & Optimization](03-quantization-optimization/)
- 🔍 [Kategorie 4: RAG Concepts](04-rag-concepts/)
- 🎓 [Kategorie 5: LLM Training](05-llm-training/)
- 📊 [Kategorie 6: Evaluation Metrics](06-evaluation-metrics/)

---

*Zentrale Definitionen für 28 Kernbegriffe – konsistent, verlinkt, fokussiert.*
