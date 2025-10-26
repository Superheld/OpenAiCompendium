# Chunking / Chunk Size / Text Segmentation

## Quick Definition

Aufteilung langer Dokumente in kleinere Segmente (Chunks) für effizientes Retrieval und LLM-Verarbeitung - kritisch für RAG-Qualität.

**Kategorie:** RAG Concepts
**Schwierigkeit:** Beginner (Konzept), Intermediate (Optimization)
**Aliases:** Chunking, Chunk Size, Text Segmentation, Document Splitting

---

## 🧠 Warum Chunking?

**Problem:** Dokumente sind zu lang für Embeddings und Context Windows!

```
Dokument: 50 Seiten Handbuch (50k tokens)
Embedding-Modell: Max 512 tokens
Context Window: 128k tokens (aber optimal: kurze, fokussierte Chunks)

→ Muss gesplittet werden!
```

**Lösung: Chunking**
```
50-Seiten Dokument
→ Split in 100 Chunks à 500 tokens
→ Jeder Chunk = 1 Embedding
→ Bei Query: Retrieve Top-K relevante Chunks
```

---

## 💡 Key Decisions

### **1. Chunk Size** (tokens)

| Size | Pros | Cons | Use Case |
|------|------|------|----------|
| **128-256** | Präzise Retrieval | Fragmentiert, Kontext verloren | FAQ, Definitionen |
| **512-1024** | ⭐ **Sweet Spot** | Gute Balance | Allgemein |
| **2048+** | Viel Kontext | Weniger präzise Retrieval | Lange Narratives |

**Default-Empfehlung:** 512 tokens

### **2. Overlap** (tokens)

```
Chunk 1: [Token 0-512]
Chunk 2: [Token 462-974]  ← 50 tokens overlap
         ^^^^^^^^^^^
         Overlap verhindert Information-Loss an Chunk-Grenzen
```

**Typisch:** 10-20% Overlap (50-100 tokens bei 512-token Chunks)

### **3. Chunking-Strategie**

**A) Fixed-Size Chunking** (einfach)
```python
chunks = [text[i:i+512] for i in range(0, len(text), 512)]
```

**B) Semantic Chunking** (besser)
```
Split an natürlichen Grenzen:
- Absätze
- Überschriften
- Satzende
```

**C) Sentence-Window** (advanced)
```
Jeder Chunk = 1 Satz
Aber: Bei Retrieval, erweitere um ±2 Sätze für Kontext
```

---

## 💻 Code-Beispiel (Kompakt)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

def chunk_document(text, chunk_size=512, overlap=50):
    """Chunk mit Token-Awareness"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=lambda t: len(enc.encode(t)),  # Token-based!
        separators=["\n\n", "\n", ". ", " ", ""]  # Semantic boundaries
    )
    chunks = splitter.split_text(text)
    return chunks

# Test
doc = "Long document... " * 1000
chunks = chunk_document(doc, chunk_size=512, overlap=50)
print(f"Created {len(chunks)} chunks")
print(f"Chunk 0: {len(enc.encode(chunks[0]))} tokens")
```

---

## ⚠️ Common Mistakes

❌ **Character-based Chunking** → Use Token-based!
❌ **No Overlap** → Information-Loss an Grenzen
❌ **Too Small** (<256 tokens) → Fragmentiert, kein Kontext
❌ **Too Large** (>2048 tokens) → Unpräzises Retrieval
❌ **Ignoriert Semantik** → Split mitten im Satz

---

## 🎯 Zusammenfassung

**Best Practice:**
- **Chunk Size:** 512 tokens
- **Overlap:** 50-100 tokens (10-20%)
- **Strategy:** Semantic (Absätze, Sätze)
- **Length Function:** Token-based (nicht characters!)

**Trade-off:**
```
Kleinere Chunks → Präziseres Retrieval, weniger Kontext
Größere Chunks → Mehr Kontext, unpräziseres Retrieval
```

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ⬅️ [Vorheriger Begriff: RAG](01-rag.md)
