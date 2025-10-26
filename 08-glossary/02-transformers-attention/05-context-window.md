# Context Window / Sequence Length / KV-Cache

## Quick Definition

Die maximale Anzahl Tokens die ein LLM gleichzeitig verarbeiten kann - bestimmt wie viel "Kontext" das Modell "sehen" kann.

**Kategorie:** Transformers & Attention
**Schwierigkeit:** Beginner (Konzept), Intermediate (Implications)
**Aliases:** Context Window, Sequence Length, Context Length, Max Tokens

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

**Stell dir vor, ein LLM hat "Arbeitsspeicher" für Text:**

```
GPT-4: 128k tokens ≈ 96k words ≈ 200 Seiten
Claude 3.5: 200k tokens ≈ 150k words ≈ 400 Seiten
Llama 3.2: 128k tokens
```

**Was passiert bei Überschreitung?**
```
Input: 150k tokens
GPT-4 Context: 128k tokens

→ Erste 22k tokens werden "vergessen" ❌
oder
→ Error: "Context length exceeded" ❌
```

### Why It Matters - Welches Problem löst Context Window?

**Problem 1: Lange Dokumente** ❌ → ✅

**Ohne ausreichendes Context Window:**
```
Dokument: 50k tokens (300 Seiten Handbuch)
GPT-3.5: 4k Context Window

→ Kann nur 4k tokens verarbeiten (3% des Dokuments!)
→ Rest wird weggeschnitten ❌
```

**Mit großem Context Window:**
```
Claude 3.5: 200k Context
→ Gesamtes Dokument passt ✅
→ Kann über gesamtes Dokument reasonen
```

**Problem 2: RAG Chunking** ❌ → ✅

**Kleines Context Window:**
```
GPT-3.5 (4k):
- Query: 100 tokens
- System Prompt: 200 tokens
- Response: 1000 tokens
→ Remaining: 2.7k tokens für Retrieved Context

Mit Chunk-Size 512 tokens:
→ Max 5 Chunks können retrieved werden ❌
```

**Großes Context Window:**
```
GPT-4 (128k):
→ Remaining: 126k tokens
→ Kann 200+ Chunks verarbeiten ✅
```

**Problem 3: Multi-Turn Conversations**

```
Context Used:
- System Prompt: 300 tokens
- Turn 1: 500 tokens
- Turn 2: 600 tokens
- Turn 3: 800 tokens
- ...
Total: Wächst mit jedem Turn!

GPT-3.5 (4k): Nach 3-5 Turns voll
GPT-4 (128k): 100+ Turns möglich
```

### Mathematische Formalisierung

**Context Window Constraint:**

$$n_{\text{tokens}} \leq C_{\text{max}}$$

wo $C_{\text{max}}$ = Max Context Window (modell-abhängig)

**Attention Complexity:**

$$\text{Memory} = O(n^2) \quad \text{(Quadratisch!)}$$

**KV-Cache Memory (während Inference):**

$$\text{KV-Cache Memory} = 2 \times n \times d_{\text{model}} \times \text{num\_layers} \times \text{batch\_size}$$

**Beispiel (Llama 3.1 8B):**
```
n = 128k tokens (context)
d_model = 4096
num_layers = 32
batch_size = 1

KV-Cache = 2 × 128k × 4096 × 32 × 1 × 2 bytes (FP16)
         = 64 GB! ← Nur für Context-Speicherung!
```

### Common Variations

**Model Context Windows (2025):**

| Model | Context Window | ≈ Pages | Use Case |
|-------|----------------|---------|----------|
| **GPT-3.5** | 4k-16k | 3-12 | Chat |
| **GPT-4** | 128k | 200 | Dokument-Analyse |
| **Claude 3.5 Sonnet** | 200k | 400 | Lange Dokumente |
| **Llama 3.2** | 128k | 200 | Open-Source |
| **Gemini 1.5 Pro** | 1M | 2000 | Extreme Long-Context |

---

## 💻 Code-Beispiel

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

# ========================================
# 1. CONTEXT WINDOW CHECK
# ========================================

def check_context_window(
    prompt,
    response_tokens=1000,
    model="gpt-4",
    context_limits={"gpt-4": 128000, "gpt-3.5-turbo": 16000}
):
    """Check ob Prompt + Response in Context passt"""
    prompt_tokens = len(enc.encode(prompt))
    total_tokens = prompt_tokens + response_tokens

    limit = context_limits[model]

    print(f"Model: {model}")
    print(f"  Prompt: {prompt_tokens} tokens")
    print(f"  Response (estimated): {response_tokens} tokens")
    print(f"  Total: {total_tokens} tokens")
    print(f"  Limit: {limit} tokens")

    if total_tokens <= limit:
        print(f"  ✅ Passt ({(total_tokens/limit)*100:.1f}% des Limits)")
    else:
        print(f"  ❌ Überschreitet Limit um {total_tokens - limit} tokens!")
        print(f"     Kürze Prompt oder reduziere Response-Länge")

# Test mit langem Dokument
long_doc = "Lorem ipsum... " * 10000  # Simuliere langes Dokument
check_context_window(long_doc, response_tokens=1000)

# ========================================
# 2. RAG CONTEXT CALCULATION
# ========================================

def calculate_rag_capacity(
    context_window=128000,
    system_prompt=300,
    query=100,
    response=2000,
    chunk_size=512
):
    """Berechne wie viele RAG Chunks passen"""
    remaining = context_window - system_prompt - query - response
    max_chunks = remaining // chunk_size

    print(f"\nRAG Capacity Calculation:")
    print(f"  Context Window: {context_window}")
    print(f"  - System Prompt: {system_prompt}")
    print(f"  - Query: {query}")
    print(f"  - Response (reserved): {response}")
    print(f"  = Remaining: {remaining}")
    print(f"  → Max Chunks ({chunk_size} tokens each): {max_chunks}")

    return max_chunks

calculate_rag_capacity(context_window=128000)  # GPT-4
calculate_rag_capacity(context_window=4000)    # GPT-3.5 (old)

# ========================================
# 3. SLIDING WINDOW FÜR LANGE DOKUMENTE
# ========================================

def process_long_document(document, context_window=128000, overlap=1000):
    """Verarbeite Dokument länger als Context Window"""
    tokens = enc.encode(document)

    if len(tokens) <= context_window:
        print(f"✅ Dokument passt in Context ({len(tokens)} tokens)")
        return [document]

    print(f"❌ Dokument zu lang ({len(tokens)} tokens)")
    print(f"   Chunking mit Sliding Window ({context_window} window, {overlap} overlap)")

    chunks = []
    for i in range(0, len(tokens), context_window - overlap):
        chunk_tokens = tokens[i:i + context_window]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

    print(f"   → {len(chunks)} Chunks erstellt")
    return chunks

very_long_doc = "Content... " * 200000
process_long_document(very_long_doc)
```

**Output:**
```
Model: gpt-4
  Prompt: 30000 tokens
  Response (estimated): 1000 tokens
  Total: 31000 tokens
  Limit: 128000 tokens
  ✅ Passt (24.2% des Limits)

RAG Capacity Calculation:
  Context Window: 128000
  - System Prompt: 300
  - Query: 100
  - Response (reserved): 2000
  = Remaining: 125600
  → Max Chunks (512 tokens each): 245

RAG Capacity Calculation:
  Context Window: 4000
  - System Prompt: 300
  - Query: 100
  - Response (reserved): 2000
  = Remaining: 1600
  → Max Chunks (512 tokens each): 3

❌ Dokument zu lang (600000 tokens)
   Chunking mit Sliding Window (128000 window, 1000 overlap)
   → 5 Chunks erstellt
```

---

## 🔗 Related Terms

### **Voraussetzungen**
- **[Token](../05-llm-training/01-token.md)**: Context Window gemessen in Tokens
- **[Self-Attention](01-self-attention.md)**: O(n²) Komplexität bestimmt Context-Limit

### **Beeinflusst**
- **[Chunking](../04-rag-concepts/02-chunking.md)**: Chunk-Size abhängig von Context Window
- **RAG**: Wie viele Chunks können retrieved werden
- **Cost**: Mehr Context = höhere API-Kosten

---

## 📍 Where This Appears

### **Primary Chapter**
- `02-modern-ai/01-llms/01-model-families.md` (Sektion 6.2) - Context Windows verschiedener Modelle
- `06-applications/04-model-selection.md` - Context Window als Selection-Criteria

### **Critical for**
- `06-applications/01-rag-systems.md` - RAG Context Capacity
- `04-advanced/02-retrieval-optimization.md` - Chunking Strategy
- API Cost Estimation

---

## ⚠️ Common Misconceptions

### ❌ "Größerer Context Window ist immer besser"
**Falsch!** Trade-offs:

| Aspekt | Kleiner Context | Großer Context |
|--------|-----------------|----------------|
| **Latenz** | Schnell | Langsam (mehr Tokens) |
| **Cost** | Günstig | Teuer (pro Token) |
| **Memory** | Wenig | Viel (KV-Cache!) |
| **Quality** | Fokussiert | "Lost in the middle" Problem |

**"Lost in the Middle":** Bei 100k Context finden LLMs Info in der Mitte schlechter als am Anfang/Ende!

### ❌ "Context Window = Working Memory"
**Teilweise wahr:** LLMs können theoretisch gesamten Context "sehen", aber:
- **Attention-Patterns** fokussieren auf relevante Teile
- **Performance degradiert** bei sehr langen Contexts
- **"Needlein Haystack" Problem**: Info in 100k Context versteckt → schwer zu finden

---

## 🎯 Zusammenfassung

**Ein Satz:** Context Window bestimmt wie viele Tokens ein LLM gleichzeitig verarbeiten kann - kritisch für lange Dokumente, RAG und Multi-Turn Chats.

**Key Takeaways:**
1. **Model-Specific**: GPT-4 (128k), Claude 3.5 (200k), Gemini 1.5 (1M)
2. **Quadratic Memory**: O(n²) Komplexität (KV-Cache!)
3. **RAG Capacity**: Bestimmt wie viele Chunks retrieved werden können
4. **Cost-Relevant**: Mehr Tokens = höhere API-Kosten

**Trade-off:**
```
Größerer Context Window → Mehr Flexibilität
                        → Aber: Höhere Kosten + Latenz
```

**Wann wichtig?**
- ✅ Lange Dokumente (Handbücher, Bücher)
- ✅ RAG mit vielen Chunks
- ✅ Multi-Turn Conversations
- ✅ Code-Analyse (große Codebases)

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ⬅️ [Vorheriger Begriff: Self-Attention](01-self-attention.md)
