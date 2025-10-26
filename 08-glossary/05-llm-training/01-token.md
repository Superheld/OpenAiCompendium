# Token / Tokenization

## Quick Definition

Die atomare Einheit die LLMs verarbeiten - Wörter werden in Sub-Word-Tokens aufgeteilt (z.B. "Laborkühlschrank" → 4 Tokens).

**Kategorie:** LLM Training
**Schwierigkeit:** Beginner
**Aliases:** Token, Tokenization, Subword Tokenization, BPE (Byte Pair Encoding)

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

**LLMs verstehen keine Wörter - sie verstehen Tokens!**

**Beispiel:**
```
Text: "Laborkühlschrank"

Tokenization:
"Labor|kühl|schrank" → 3 Tokens

Nicht:
"Laborkühlschrank" → 1 Wort (zu komplex!)
"L|a|b|o|r|..." → 18 Buchstaben (zu granular!)
```

**Token = Sweet Spot zwischen Zeichen und Wörtern**

**Warum?**
- **Häufige Wörter** = 1 Token: "the", "ist", "und"
- **Seltene Wörter** = mehrere Tokens: "Laborkühlschrank" → 3 Tokens
- **Neue Wörter** = Kombination: "ChatGPT" → "Chat" + "GPT"

### Mathematische Formalisierung

**Tokenization-Funktion:**
$$T: \text{Text} \rightarrow [\text{token}_1, \text{token}_2, \ldots, \text{token}_n]$$

**Beispiel:**
$$T(\text{"Der Kühlschrank ist defekt"}) = [393, 24998, 1124, 84003]$$

Jedes Token → Token-ID (Integer)

**Vocabulary Size:**
$$|V| = \text{Anzahl unique Tokens}$$

Typische Vocab-Sizes:
- GPT-3/4: 100k Tokens
- Llama 3: 128k Tokens
- Claude: 200k+ Tokens

**Token-Länge vs. Text-Länge:**
$$\text{Tokens} \approx 0.75 \times \text{Words} \quad \text{(English)}$$
$$\text{Tokens} \approx 1.0 \times \text{Words} \quad \text{(German, mehr compound words)}$$

### Why It Matters - Welches Problem lösen Tokens?

**Problem 1: Vocabulary Explosion** ❌ → ✅

**Ohne Tokenization (Word-Level):**
```
Vocabulary: Jedes unique Wort = 1 Entry
Deutsch: ~300,000 Wörter
+ Komposita: "Labor|kühl|schrank", "Kühl|schrank|tür", ...
→ Millionen Vocabulary-Entries! ❌
```

**Mit Tokenization (Subword):**
```
Vocabulary: ~100k Tokens
"Laborkühlschrank" = "Labor" + "kühl" + "schrank"
→ Kann JEDES Wort aus Subwords bauen! ✅
```

**Problem 2: Rare Words** ❌ → ✅

**Word-Level:**
```
Training: "Kühlschrank" gesehen (50× im Trainingsdata)
Inference: "Laborkühlschrank" → <UNK> (unknown!) ❌
```

**Subword Tokenization:**
```
Training: "Labor" (1000×), "kühl" (800×), "schrank" (500×)
Inference: "Laborkühlschrank" → "Labor" + "kühl" + "schrank" ✅
```

**Problem 3: Cost & Context Window** 💰

**Token-Counting = Cost:**
```
GPT-4 API:
- Input:  $0.03 per 1k tokens
- Output: $0.06 per 1k tokens

Query: "Erkläre Quantization" (3 tokens)
Response: 500 tokens
Cost: 3 × $0.03/1000 + 500 × $0.06/1000 = $0.03
```

**Context Window:**
```
GPT-4: 128k tokens
≈ 96k words (English)
≈ 100 pages of text
```

### Common Variations

**1. Byte Pair Encoding (BPE)** - GPT, Llama
```
Algorithmus:
1. Start: Einzelne Zeichen
2. Finde häufigstes Zeichen-Paar
3. Merge zu neuem Token
4. Repeat bis Vocab-Size erreicht
```

**2. WordPiece** - BERT
```
Similar zu BPE, aber likelihood-basiert
"##" Prefix für Subwords: "playing" → "play" + "##ing"
```

**3. Unigram** - SentencePiece (verwendet von T5)
```
Probabilistic: behält wahrscheinlichste Subwords
```

---

## 💻 Code-Beispiel

```python
from transformers import AutoTokenizer

# ========================================
# 1. TOKENIZATION VERSTEHEN
# ========================================

# GPT-4 Tokenizer (tiktoken)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")

text = "Der Laborkühlschrank ist defekt."

# Tokenize
tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Anzahl Tokens: {len(tokens)}")

# Decode zurück
decoded = enc.decode(tokens)
print(f"Decoded: {decoded}")

# Token-by-Token
for token_id in tokens:
    token_str = enc.decode([token_id])
    print(f"  {token_id:5d} → '{token_str}'")

# ========================================
# 2. TOKEN-COUNTING FÜR COST-ESTIMATION
# ========================================

def estimate_api_cost(prompt, response_tokens, model="gpt-4"):
    """Schätze API-Kosten"""
    prompt_tokens = len(enc.encode(prompt))

    costs = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1k tokens
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    }

    cost_input = (prompt_tokens / 1000) * costs[model]["input"]
    cost_output = (response_tokens / 1000) * costs[model]["output"]
    total_cost = cost_input + cost_output

    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "cost": total_cost
    }

prompt = "Erkläre Quantization im Detail mit Beispielen."
response_len = 500  # erwartete Response-Länge

costs = estimate_api_cost(prompt, response_len)
print(f"\nCost Estimation:")
print(f"  Input:  {costs['prompt_tokens']} tokens")
print(f"  Output: {costs['response_tokens']} tokens")
print(f"  Total:  {costs['total_tokens']} tokens")
print(f"  Cost:   ${costs['cost']:.4f}")

# ========================================
# 3. CONTEXT WINDOW CHECK
# ========================================

def check_context_window(text, model="gpt-4", context_limit=128000):
    """Check ob Text in Context Window passt"""
    tokens = enc.encode(text)

    if len(tokens) <= context_limit:
        print(f"✅ {len(tokens)} tokens (passt in {context_limit} Context Window)")
    else:
        print(f"❌ {len(tokens)} tokens (überschreitet {context_limit} Limit!)")
        print(f"   Kürze um {len(tokens) - context_limit} tokens")

long_text = "Lorem ipsum... " * 10000
check_context_window(long_text)

# ========================================
# 4. CHUNKING MIT TOKEN-AWARENESS
# ========================================

def chunk_by_tokens(text, max_tokens=512, overlap=50):
    """Chunk Text basierend auf Token-Count"""
    tokens = enc.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

text = "Very long document... " * 1000
chunks = chunk_by_tokens(text, max_tokens=512)
print(f"\nChunked into {len(chunks)} chunks")
```

**Output:**
```
Text: Der Laborkühlschrank ist defekt.
Tokens: [9663, 9832, 79388, 17704, 42880, 31866, 374, 711, 36634, 13]
Anzahl Tokens: 10

  9663 → 'Der'
  9832 → ' Labor'
 79388 → 'k'
 17704 → 'ühl'
 42880 → 'schr'
 31866 → 'ank'
   374 → ' ist'
   711 → ' def'
 36634 → 'ekt'
    13 → '.'

Cost Estimation:
  Input:  9 tokens
  Output: 500 tokens
  Total:  509 tokens
  Cost:   $0.0303

✅ 30000 tokens (passt in 128000 Context Window)

Chunked into 59 chunks
```

---

## 🔗 Related Terms

### **Verwendet Tokenization**
- **Context Window**: Max Tokens ein LLM verarbeiten kann
- **Chunking**: Token-aware Chunking für RAG
- **Cost Estimation**: API-Kosten basieren auf Tokens

### **Verwandt**
- **Vocabulary**: Menge aller Tokens (~100k-200k)
- **Byte Pair Encoding (BPE)**: Tokenization-Algorithmus
- **Embedding**: Jedes Token wird zu Embedding

---

## 📍 Where This Appears

### **Primary Chapter**
- `02-modern-ai/01-llms/01-model-families.md` (Sektion 1.2) - Token-Counting, Context Windows
- `03-core/01-training/` - Tokenization im Training

### **Critical for**
- `06-applications/01-rag-systems.md` - Chunking mit Token-Awareness
- `04-advanced/02-retrieval-optimization.md` - Chunk-Size in Tokens
- API Cost Estimation (alle LLM APIs)

---

## ⚠️ Common Misconceptions

### ❌ "1 Token = 1 Wort"
**Falsch!** Häufige Wörter = 1 Token, seltene = mehrere:

```
"Der" → 1 Token
"Laborkühlschrank" → 3-4 Tokens
"Supercalifragilisticexpialidocious" → 8+ Tokens
```

**Rule of thumb:**
- English: ~0.75 tokens per word
- German: ~1.0 tokens per word (mehr Komposita!)

### ❌ "Tokenization ist bei allen Modellen gleich"
**Falsch!** Jedes Modell hat eigenen Tokenizer:

```python
gpt4_tokens = tiktoken.encode("Test")     # [2323]
llama_tokens = llama_tokenizer("Test")    # [12345]

# Unterschiedlich! Nicht kompatibel!
```

**Problem:** Context Window Limits sind modell-spezifisch!

### ❌ "Token-Count ist egal"
**SEHR FALSCH!** Token-Count bestimmt:
- **API-Kosten**: $0.03-0.06 per 1k tokens
- **Context Window**: Überschreitung → Fehler
- **Latenz**: Mehr Tokens = langsamere Response
- **Chunking**: Chunk-Size sollte in Tokens sein, nicht Characters!

---

## 🎯 Zusammenfassung

**Ein Satz:** Tokens sind Subword-Einheiten (zwischen Zeichen und Wörtern), die LLMs verarbeiten - kritisch für Cost-Estimation und Context-Windows.

**Formel (Approximation):**
$$\text{Tokens} \approx 0.75 \times \text{Words} \quad \text{(English)}$$

**Key Takeaways:**
1. **Subword-Level**: "Laborkühlschrank" → 3-4 Tokens
2. **Cost-Relevant**: API-Kosten basieren auf Token-Count
3. **Context-Window**: GPT-4 = 128k tokens ≈ 100 pages
4. **Model-Specific**: Jedes Modell hat eigenen Tokenizer

**Warum Tokens?**
→ Löst Vocabulary-Explosion + Rare-Words Problem + ermöglicht exakte Cost-Berechnung! 🚀

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ➡️ [Nächster Begriff: Fine-Tuning](03-fine-tuning.md)
