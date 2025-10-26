# Quantization / Model Quantization

## Quick Definition

Reduzierung der numerischen Präzision von Modell-Gewichten (z.B. 16-bit → 4-bit), um Memory und Compute zu sparen - mit minimalem Qualitätsverlust.

**Kategorie:** Quantization & Optimization
**Schwierigkeit:** Intermediate
**Aliases:** Quantization, Model Quantization, Weight Quantization

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

**Problem:** Llama 3.1 70B hat 70 Milliarden Parameter. Mit FP16 (2 bytes pro Parameter):
```
70B × 2 bytes = 140 GB Memory
```
→ **Unmöglich** auf Consumer-GPUs (24-48 GB)!

**Lösung: Quantization**
Speichere jedes Gewicht mit weniger Bits:

| Präzision | Bytes/Param | Llama 70B Memory | Qualität |
|-----------|-------------|------------------|----------|
| **FP32** | 4 | 280 GB | Baseline |
| **FP16** | 2 | 140 GB | ~identisch |
| **INT8** | 1 | 70 GB | 99% ✅ |
| **INT4** | 0.5 | 35 GB | 95-98% ✅ |
| **INT2** | 0.25 | 17.5 GB | <90% ❌ |

**Beispiel:**
```
Original (FP16):  3.14159265 (16 bits)
Quantized (INT4): 3.1 (4 bits) ← 4× weniger Memory!
```

### Mathematische Formalisierung

**Quantization-Funktion:**

$$Q(w) = \text{round}\left(\frac{w - w_{\min}}{w_{\max} - w_{\min}} \times (2^b - 1)\right)$$

wo:
- $w$ = Original-Gewicht (FP16/FP32)
- $b$ = Bit-Tiefe (4 für INT4, 8 für INT8)
- $[w_{\min}, w_{\max}]$ = Wertebereich

**De-Quantization (für Inference):**

$$\hat{w} = \frac{Q(w)}{2^b - 1} \times (w_{\max} - w_{\min}) + w_{\min}$$

**Quantization Error:**

$$\epsilon = |w - \hat{w}|$$

**Ziel:** Minimiere $\epsilon$ über alle Gewichte.

### Why It Matters - Welches Problem löst Quantization?

**Problem 1: Memory-Constraints** ❌ → ✅

**Ohne Quantization:**
```
Llama 3.1 8B (FP16):  16 GB RAM
Llama 3.1 70B (FP16): 140 GB RAM ← Braucht 6× A100 GPUs!
```

**Mit Quantization (INT4):**
```
Llama 3.1 8B (INT4):  4 GB RAM  ← Läuft auf Consumer-GPU!
Llama 3.1 70B (INT4): 35 GB RAM ← Läuft auf 2× RTX 4090!
```

**Problem 2: Inference-Speed** ❌ → ✅

Weniger Bits = weniger Memory Bandwidth = schneller:

| Quantization | Memory | Tokens/sec (Llama 8B) |
|--------------|--------|-----------------------|
| FP16 | 16 GB | 25 tok/s |
| INT8 | 8 GB | 40 tok/s (+60%) |
| INT4 | 4 GB | 65 tok/s (+160%) |

**Problem 3: Deployment-Kosten** ❌ → ✅

```
Cloud GPU Costs (pro Stunde):
- A100 80GB (FP16):  $3-4/h
- RTX 4090 (INT4):   $0.50/h  ← 6-8× billiger!
```

### Common Variations

**1. GGUF (llama.cpp Format)**
- Für CPU-Inference optimiert
- Verschiedene Quantization-Schemas:
  - **Q4_K_M**: 4-bit K-Quant Medium (empfohlen)
  - **Q8_0**: 8-bit (höchste Qualität)
  - **Q2_K**: 2-bit (experimentell, niedrige Qualität)

**Dateiname-Schema:**
```
llama-3.1-8b-instruct.Q4_K_M.gguf
                      ^^^^^^^^
                      Quantization-Schema
```

**2. GPTQ (GPU-optimized)**
- Für Nvidia GPUs optimiert
- 4-bit Quantization
- Activations bleiben FP16 (nur Gewichte quantisiert)

**3. AWQ (Activation-aware Weight Quantization)**
- Intelligenter als GPTQ: gewichtet wichtige Gewichte höher
- Perplexity-Vergleich auf WikiText-2:
  - FP16: 5.47
  - GPTQ: 5.54 (+1.3% schlechter)
  - AWQ: 5.50 (+0.5% schlechter) ✅ Besser!

**4. bitsandbytes (QLoRA)**
- 8-bit (LLM.int8()): Fast kein Qualitätsverlust
- 4-bit (QLoRA): Für Fine-Tuning optimiert

---

## 💻 Code-Beispiel

```python
# ========================================
# 1. MEMORY-BERECHNUNG
# ========================================

def calculate_model_memory(num_params_billions, precision):
    """Berechne Memory-Footprint eines LLMs"""
    bytes_per_param = {
        'FP32': 4,
        'FP16': 2,
        'INT8': 1,
        'INT4': 0.5,
    }

    memory_gb = num_params_billions * bytes_per_param[precision]
    return memory_gb

# Llama 3.1 Beispiele
models = [
    ('Llama 3.1 8B', 8),
    ('Llama 3.1 70B', 70),
    ('Llama 3.1 405B', 405),
]

print("Model Memory Requirements:\n")
for name, size in models:
    print(f"{name}:")
    for prec in ['FP16', 'INT8', 'INT4']:
        mem = calculate_model_memory(size, prec)
        print(f"  {prec}: {mem:.1f} GB")
    print()

# ========================================
# 2. LADE QUANTIZED MODEL (Transformers)
# ========================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Option A: INT8 (bitsandbytes)
model_int8 = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Automatische INT8-Quantization!
    device_map="auto"
)

# Option B: INT4 (bitsandbytes / QLoRA)
model_int4 = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # INT4-Quantization
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Inference
prompt = "Erkläre Quantization in einem Satz:"
inputs = tokenizer(prompt, return_tensors="pt").to(model_int4.device)

outputs = model_int4.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ========================================
# 3. GGUF MIT LLAMA.CPP (CPU-Inference)
# ========================================

# Download GGUF model
# wget https://huggingface.co/.../llama-3.1-8b-instruct.Q4_K_M.gguf

# Run mit llama.cpp
# ./llama-cli -m llama-3.1-8b-instruct.Q4_K_M.gguf -p "Prompt"

# Python API (llama-cpp-python)
from llama_cpp import Llama

llm = Llama(
    model_path="llama-3.1-8b-instruct.Q4_K_M.gguf",
    n_ctx=2048,  # Context window
    n_gpu_layers=0  # CPU-only
)

output = llm(
    "Erkläre Quantization:",
    max_tokens=50,
    temperature=0.7
)

print(output['choices'][0]['text'])
```

**Output:**
```
Model Memory Requirements:

Llama 3.1 8B:
  FP16: 16.0 GB
  INT8: 8.0 GB
  INT4: 4.0 GB

Llama 3.1 70B:
  FP16: 140.0 GB
  INT8: 70.0 GB
  INT4: 35.0 GB

Llama 3.1 405B:
  FP16: 810.0 GB
  INT8: 405.0 GB
  INT4: 202.5 GB

Erkläre Quantization in einem Satz: Quantization reduziert die
numerische Präzision von Modell-Gewichten, um Memory zu sparen.
```

---

## 🔗 Related Terms

### **Voraussetzungen**
- **[Tensor](../01-vectors-embeddings/00-tensor.md)**: Was quantisiert wird (Gewichts-Tensoren)
- **Floating Point**: FP32, FP16, BF16 Präzision
- **Integer**: INT8, INT4 Präzision

### **Verwandt**
- **LoRA/QLoRA**: Fine-Tuning mit quantisierten Modellen
- **Model Serving**: Deployment mit quantisierten Modellen
- **GPU Memory**: Constraints die Quantization nötig machen

### **Methoden**
- GGUF (llama.cpp)
- GPTQ (GPU-optimiert)
- AWQ (Activation-aware)
- bitsandbytes (QLoRA)

---

## 📍 Where This Appears

### **Primary Chapter** (wird jetzt referenziert statt 224 Zeilen!)
- `02-modern-ai/01-llms/01-model-families.md` (Sektion 5, Zeilen 1136-1360) - Vollständige Quantization-Guide
- `03-core/04-optimization/` - Optimierungs-Techniken

### **Usage Examples**
- `06-applications/04-model-selection.md` - Quantization bei Modell-Wahl
- `02-modern-ai/01-llms/01-model-families.md` - Llama, Mistral Quantization

### **Mentioned In**
- `05-ethics/05-safety-guardrails.md` - Effizienz und Environmental Impact
- `03-core/05-infrastructure/02-model-serving.md` - Deployment

---

## ⚠️ Common Misconceptions

### ❌ "Quantization ruiniert die Modell-Qualität"
**Falsch!** INT8 und INT4 haben minimalen Qualitätsverlust:

**Benchmark (Llama 3.1 8B auf MMLU):**
```
FP16:  68.4%
INT8:  68.1% (-0.3%) ← Fast identisch!
INT4:  66.8% (-1.6%) ← Akzeptabel für 4× Memory-Einsparung
INT2:  42.3% (-26%) ← Zu aggressiv!
```

**Richtig:** INT4/INT8 sind sweet spot für Production.

### ❌ "Quantization ist nur für Inference"
**Falsch!** QLoRA ermöglicht Fine-Tuning mit quantisierten Modellen:

```python
# Fine-Tune Llama 70B auf 1× RTX 4090 (24GB)!
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    load_in_4bit=True,  # Quantization
    # + LoRA (Low-Rank Adaptation)
)
```

**Ohne Quantization:** Llama 70B Fine-Tuning braucht 280GB (unmöglich auf Consumer-GPU)
**Mit QLoRA:** Läuft auf 24GB GPU ✅

### ❌ "Alle Quantization-Methoden sind gleich"
**Falsch!** Große Unterschiede:

| Methode | Best for | Perplexity | Speed |
|---------|----------|------------|-------|
| **GGUF** | CPU Inference | Mittel | Langsam (CPU) |
| **GPTQ** | GPU Inference | Gut | Schnell (GPU) |
| **AWQ** | GPU Inference | **Best** | Schnell (GPU) |
| **bitsandbytes** | Training (QLoRA) | Gut | Mittel |

**Richtig:** AWQ hat beste Qualität, GGUF für CPU, bitsandbytes für Training.

---

## 📊 Quantization Comparison Table

| Aspect | FP16 | INT8 | INT4 | INT2 |
|--------|------|------|------|------|
| **Memory** | Baseline | 2× weniger | 4× weniger | 8× weniger |
| **Speed** | Baseline | +60% | +160% | +300% |
| **Quality** | 100% | 99% | 95-98% | <90% |
| **Use Case** | Training | Production | Consumer GPUs | Experimentell |
| **Llama 8B** | 16 GB | 8 GB | 4 GB | 2 GB |
| **Llama 70B** | 140 GB | 70 GB | 35 GB | 17.5 GB |

**Empfehlung:**
- **Training**: FP16 oder INT8
- **Production (GPU)**: INT8 (GPTQ/AWQ)
- **Consumer Hardware**: INT4 (GGUF)
- **Experimentell**: INT2 (nicht für Production!)

---

## 🎯 Zusammenfassung

**Ein Satz:** Quantization reduziert Bit-Präzision von Gewichten (FP16 → INT4), um Memory 4× zu reduzieren mit nur 2-5% Qualitätsverlust.

**Formel (Memory-Berechnung):**
$$\text{Memory (GB)} = \text{Params (Billions)} \times \text{Bytes per Param}$$

**Key Takeaways:**
1. **Memory-Reduktion**: INT4 = 4× weniger als FP16
2. **Speed-Boost**: +60% (INT8) bis +160% (INT4)
3. **Minimaler Qualitätsverlust**: INT8 (~1%), INT4 (~2-5%)
4. **Enables Consumer GPUs**: Llama 70B INT4 läuft auf 2× RTX 4090

**Trade-off:**
```
Memory ↓↓  vs.  Quality ↓ (minimal!)
```

**Wann nutzen?**
- ✅ Consumer GPUs (24-48 GB)
- ✅ Production Inference (Cost-Saving)
- ✅ Edge Deployment (Mobile, Embedded)
- ❌ Training (nur mit QLoRA)

**Das Problem das Quantization löst:** Macht 70B-405B Parameter Modelle auf Consumer-Hardware lauffähig! 🚀

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- 🏠 [Zurück zum Glossar](../00-overview.md)
- ➡️ [Nächste Kategorie: RAG Concepts](../04-rag-concepts/)
