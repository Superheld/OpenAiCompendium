# Transformers & Attention: Glossar

## 🎯 Übersicht

Diese Kategorie definiert **Transformer-Architektur und Attention-Mechanismen** - die Revolution, die GPT, BERT und alle modernen LLMs ermöglicht hat.

**Kernfrage:** Wie funktionieren Transformer und warum sind sie so mächtig?

---

## 📋 Begriffe in dieser Kategorie

### **Kern-Mechanismen (2 Begriffe)**
1. **[01-self-attention.md](01-self-attention.md)** - Self-Attention / Attention Mechanism
   - **KRITISCH**: Das fundamentale Konzept hinter Transformers
   - Query, Key, Value (Q, K, V)
   - Warum "Attention is All You Need"

2. **[02-multi-head-attention.md](02-multi-head-attention.md)** - Multi-Head Attention
   - Warum mehrere "Köpfe"?
   - Parallel lernen verschiedener Repräsentationen
   - Wie viele Heads? (z.B. GPT-3: 96 Heads)

### **Position & Kontext (2 Begriffe)**
3. **[03-positional-encoding.md](03-positional-encoding.md)** - Positional Encoding / RoPE / ALiBi
   - Problem: Transformer hat keine "Reihenfolge"
   - Sinusoidal, Learned, RoPE, ALiBi
   - Warum wichtig für Sprachverständnis

4. **[05-context-window.md](05-context-window.md)** - Context Window / Sequence Length / KV-Cache
   - **KRITISCH**: Max Tokens ein LLM verarbeiten kann
   - Trade-offs: GPT-4 (128k) vs. Llama 3.2 (128k) vs. Claude 3.5 (200k)
   - KV-Cache Memory Kosten

### **Architektur (1 Begriff)**
5. **[04-transformer-block.md](04-transformer-block.md)** - Transformer Block / Transformer Layer
   - Aufbau: Attention + Feed-Forward + Normalization
   - Encoder vs. Decoder Blocks
   - Wie viele Layers? (GPT-3: 96 Layers)

---

## 🔗 Lernpfad: Empfohlene Reihenfolge

```
1. Self-Attention (01) → verstehe das Kern-Konzept
   ↓
2. Multi-Head Attention (02) → warum parallel mehrere Attention-Mechanismen?
   ↓
3. Positional Encoding (03) → wie Transformer Reihenfolge versteht
   ↓
4. Transformer Block (04) → wie alles zusammenkommt
   ↓
5. Context Window (05) → Grenzen und Trade-offs
```

---

## 🎓 Was du danach kannst

Nach Durcharbeiten dieser 5 Begriffe kannst du:

- ✅ **Erklären** wie Self-Attention mathematisch funktioniert (Q·K^T)
- ✅ **Verstehen** warum Transformer paralleler sind als RNNs
- ✅ **Berechnen** Memory-Kosten von KV-Cache
- ✅ **Entscheiden** zwischen RoPE vs. ALiBi Positional Encoding
- ✅ **Abschätzen** wie viele Layers/Heads für ein Problem nötig sind

---

## 🔗 Verwandte Themen im Kompendium

### **Historischer Kontext:**
- `01-historical/04-attention-transformers/` - "Attention is All You Need" (2017)
- `01-historical/03-deep-learning/` - Warum Transformer RNNs ablösten

### **Moderne Anwendung:**
- `02-modern-ai/01-llms/01-model-families.md` - Transformer in GPT, Llama, Mistral
- `02-modern-ai/02-vision/` - Vision Transformers (ViT)

### **Praktische Implementierung:**
- `06-applications/04-model-selection.md` - Context Window bei Modell-Wahl
- `03-core/01-training/` - Training von Transformers

### **Advanced Topics:**
- `04-advanced/` - Flash Attention, Sparse Attention, Long-Context Techniques

---

## 📊 Cross-Reference Matrix

| Begriff | Verwendet in | Voraussetzung für |
|---------|--------------|-------------------|
| **Self-Attention** | 15 Dateien | Alle Transformer-Modelle |
| **Multi-Head Attention** | 12 Dateien | GPT, BERT, alle modernen LLMs |
| **Positional Encoding** | 8 Dateien | Sprachverständnis, Reihenfolge |
| **Context Window** | 13 Dateien | RAG Chunking, Cost Estimation |
| **Transformer Block** | 10 Dateien | Architektur-Verständnis |

---

**Navigation:**
- 🏠 [Zurück zum Glossar](../00-overview.md)
- ⬅️ [Vorige Kategorie: Vectors & Embeddings](../01-vectors-embeddings/)
- ➡️ [Nächste Kategorie: Quantization & Optimization](../03-quantization-optimization/)
