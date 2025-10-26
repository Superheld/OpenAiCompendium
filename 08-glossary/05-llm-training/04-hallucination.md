# Hallucination / Factual Grounding

## Quick Definition

LLMs "erfinden" Fakten die plausibel klingen aber faktisch falsch sind - das zentrale Qualitätsproblem für Production-Deployments.

**Kategorie:** LLM Training & Ethics
**Schwierigkeit:** Beginner (Konzept), Advanced (Mitigation)
**Aliases:** Hallucination, Factual Grounding, Faithfulness, Fabrication

---

## 🧠 Was sind Hallucinations?

**Definition:** LLM generiert Output der:
1. **Plausibel klingt** (grammatisch korrekt, kohärent)
2. **Faktisch falsch ist** (nicht in Training-Data oder Context)
3. **Confident präsentiert wird** (keine Unsicherheit signalisiert)

---

## 💡 Hallucination-Typen

### **1. Factual Hallucination** (Fakten-Erfindung)

```
User: "Wer hat Quantencomputer erfunden?"
LLM:  "Richard Feynman erfand 1982 den ersten Quantencomputer."
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      FALSCH! (Konzept: Feynman 1982, aber kein "Computer" gebaut)
```

### **2. Contextual Hallucination** (Ignoriert Context)

```
Context: "Laborkühlschrank LK-42 funktioniert einwandfrei."
Query:   "Ist LK-42 defekt?"
LLM:     "Ja, LK-42 ist seit letzter Woche defekt."
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         FALSCH! Ignoriert Context (sagt "funktioniert")
```

### **3. Extrinsic Hallucination** (Erfundene Details)

```
User: "Zusammenfassung des Meetings?"
LLM:  "Meeting am 15.01.2025 um 14:00 Uhr. Teilnehmer: Anna, Bob, Carl."
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      ERFUNDEN! (Keine dieser Details im Context)
```

---

## 💻 Detection & Mitigation

### **Detection: Selbst-Konsistenz Check**

```python
def detect_hallucination_selfconsistency(llm, query, n=5, threshold=0.7):
    """Generate multiple responses, check consistency"""
    responses = [llm.generate(query) for _ in range(n)]

    # Compare responses (simplified)
    similarities = compute_pairwise_similarity(responses)
    avg_similarity = np.mean(similarities)

    if avg_similarity < threshold:
        return "POSSIBLE HALLUCINATION (inconsistent responses)"
    else:
        return "LIKELY FACTUAL (consistent)"
```

### **Mitigation: RAG (Best Solution)**

```python
# Without RAG (Hallucination Risk!)
response = llm.generate("Ist LK-42 defekt?")

# With RAG (Grounded!)
retrieved_docs = vector_db.search("LK-42 Status")
context = "\n".join(retrieved_docs)
prompt = f"Context: {context}\n\nQuestion: Ist LK-42 defekt?"
response = llm.generate(prompt)
```

**RAG reduziert Hallucinations um 40-60%!**

### **Mitigation: Instruction Prompting**

```python
system_prompt = """You are a factual assistant.
Rules:
1. ONLY answer based on provided context
2. If unsure, say "I don't have enough information"
3. NEVER make up facts
4. Cite sources when possible
"""
```

### **Mitigation: Confidence Scoring**

```python
# Ask LLM to rate its own confidence
prompt = f"{query}\n\nConfidence (0-100): "
response = llm.generate(prompt)
# "Answer: X. Confidence: 65"

if confidence < 70:
    response += "\n⚠️ Note: Low confidence, verify independently"
```

---

## 📊 Hallucination Benchmark (TruthfulQA)

| Model | Truthfulness | Hallucination Rate |
|-------|--------------|---------------------|
| **GPT-3.5** | 47% | 53% ❌ |
| **GPT-4** | 59% | 41% |
| **GPT-4 + RAG** | 78% | 22% ✅ |
| **Claude 3** | 75% | 25% |

**Takeaway:** Even SOTA models hallucinate 20-40%!

---

## 🔗 Related Terms

### **Solutions**
- **[RAG](../04-rag-concepts/01-rag.md)**: Beste Mitigation (external grounding)
- **Fine-Tuning**: Train auf factual accuracy
- **RLHF**: Reward truthfulness

### **Evaluation**
- **Faithfulness**: Antworttreue zum Context
- **Factuality**: Faktische Korrektheit
- **Attribution**: Source-Citation

---

## 📍 Where This Appears

### **Primary Chapter**
- `03-core/03-evaluation/02-ai-evaluation/06-hallucination-detection.md` - Detection & Metrics
- `05-ethics/05-safety-guardrails.md` - Safety Implications

---

## ⚠️ Why Hallucinations Happen

**1. Training Objective Mismatch**
```
Training: "Generate plausible next token"
NOT: "Generate factually correct token"

→ Plausibilität ≠ Faktualität!
```

**2. No Access to External Knowledge**
```
LLM: Only Training Data (cutoff date!)
→ Keine aktuellen/spezifischen Daten
→ "Füllt Lücken" mit Plausiblem
```

**3. Overgeneralization**
```
Training: Sah viele "X verursacht Y" Beispiele
→ Generalisiert: "X IMMER verursacht Y"
→ Aber: Ausnahmen existieren!
```

---

## 🎯 Zusammenfassung

**Problem:** LLMs "halluzinieren" = erfinden plausible aber falsche Fakten.

**Mitigation-Strategien:**
1. **RAG** (Best!) → External Knowledge Base
2. **Instruction Prompting** → "Only use context, don't make up facts"
3. **Confidence Scoring** → Flag low-confidence outputs
4. **Self-Consistency** → Generate multiple times, check consistency
5. **Fine-Tuning** → Train auf factual accuracy

**Key Takeaway:**
- Hallucinations sind NICHT bug sondern feature (Training-Objective!)
- Selbst GPT-4 halluziniert 40%
- **RAG reduziert auf ~20%** ✅

**Production-Kritisch:**
```
Medical, Legal, Financial → MUSS Hallucinations detektieren/mitigieren!
```

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ⬅️ [Vorheriger Begriff: Fine-Tuning](03-fine-tuning.md)
