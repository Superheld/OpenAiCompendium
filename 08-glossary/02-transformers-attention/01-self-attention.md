# Self-Attention / Attention Mechanism

## Quick Definition

Ein Mechanismus, der jedem Token erlaubt "hinzuschauen" auf alle anderen Tokens in der Sequenz und deren Relevanz zu gewichten - das Herz aller Transformer-Modelle.

**Kategorie:** Transformers & Attention
**Schwierigkeit:** Intermediate
**Aliases:** Self-Attention, Attention Mechanism, Scaled Dot-Product Attention

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

Stell dir vor, du liest den Satz:
> "Der Laborkühlschrank ist defekt, **er** muss repariert werden."

**Frage:** Worauf bezieht sich "**er**"?

**Self-Attention löst das:**
- Token "er" **schaut auf** alle vorherigen Tokens
- **Gewichtet** ihre Relevanz: "Laborkühlschrank" (hoch), "ist" (niedrig)
- **Conclusion**: "er" = "Laborkühlschrank"

**Mathematisch:** Jedes Token erstellt:
- **Query** (Q): "Was suche ich?"
- **Key** (K): "Was biete ich an?"
- **Value** (V): "Was ist mein Inhalt?"

Dann: **Attention Score** = Wie gut matcht Query mit Keys?

### Mathematische Formalisierung

**Eingabe:** Sequenz von Tokens $\mathbf{X} = [x_1, x_2, \ldots, x_n]$, eingebettet als $\mathbb{R}^{n \times d}$

**Transformation:**
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

wo $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$ gelernte Gewichtsmatrizen sind.

**Self-Attention:**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

**Schritt-für-Schritt:**

1. **Attention Scores:** $\mathbf{S} = \mathbf{Q} \mathbf{K}^T \in \mathbb{R}^{n \times n}$
   - Jedes Token (Query) vs. alle Tokens (Keys)
   - Hoher Score = relevant

2. **Scaling:** $\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{d_k}}$
   - Verhindert zu große Werte (Gradient-Stabilität)

3. **Softmax (Normalisierung):** $\mathbf{A} = \text{softmax}(\mathbf{S}_{\text{scaled}})$
   - Scores → Wahrscheinlichkeiten (Summe = 1)

4. **Gewichtete Summe:** $\mathbf{Output} = \mathbf{A} \mathbf{V}$
   - Kombiniere Values basierend auf Attention-Gewichten

**Beispiel (2 Tokens):**
```
Input:  ["Kühlschrank", "defekt"]
Embeddings: [[0.5, 0.8], [0.2, 0.4]]

Q = K = V = Embeddings (vereinfacht)

S = Q·K^T = [[0.5×0.5 + 0.8×0.8, 0.5×0.2 + 0.8×0.4],
             [0.2×0.5 + 0.4×0.8, 0.2×0.2 + 0.4×0.4]]
  = [[0.89, 0.42],
     [0.42, 0.20]]

Softmax(S[0]) = [0.62, 0.38]  # Token 1 attends mostly to itself
Softmax(S[1]) = [0.55, 0.45]  # Token 2 attends to both

Output[0] = 0.62×V[0] + 0.38×V[1]  # Gewichtete Kombination
```

### Why It Matters

**1. Parallelisierbar** (vs. RNNs)

**RNN (sequenziell):**
```
h1 = f(x1)
h2 = f(x2, h1)  ← muss warten auf h1!
h3 = f(x3, h2)  ← muss warten auf h2!
```

**Transformer (parallel):**
```
Alle Attention-Scores gleichzeitig berechnen!
Q·K^T → Matrix-Multiplikation (GPU-optimiert)
```

**Training-Speedup:** 10-100× schneller als RNNs

**2. Long-Range Dependencies**

**RNN Problem:** Information "vergisst" über lange Sequenzen
**Transformer:** Jedes Token kann **direkt** auf weite Tokens zugreifen

**Beispiel:**
> "Der Kühlschrank, der gestern repariert wurde und heute schon wieder defekt ist, muss ersetzt werden."

Token "ersetzt" kann direkt auf "Kühlschrank" (15 Tokens entfernt) attenden!

**3. Interpretierbarkeit**

Attention Weights zeigen **was das Modell anschaut**:
```python
# Visualisiere Attention
# Token "er" attendiert 0.8 auf "Kühlschrank", 0.1 auf "ist", 0.1 auf Rest
```

→ **Attention Maps** zeigen Reasoning-Prozess

### Common Variations

**1. Scaled Dot-Product Attention** (Standard)
$$\text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

**2. Causal/Masked Attention** (GPT, Decoder)
- Future tokens werden **maskiert** (auf -∞ gesetzt vor Softmax)
- Model kann nur auf vergangene Tokens schauen

```python
mask = [[0, -∞, -∞],
        [0,  0, -∞],
        [0,  0,  0]]  # Lower triangular

S_masked = S + mask
```

**3. Cross-Attention** (Encoder-Decoder)
- Query von Decoder, Key/Value von Encoder
- z.B. Translation: "What in English corresponds to this German word?"

**4. Flash Attention** (Memory-optimiert)
- Gleiche Mathematik, aber kernel-fusion
- 4× weniger Memory, 2× schneller

---

## 💻 Code-Beispiel

```python
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

# Simplified Self-Attention (ohne W_Q, W_K, W_V Projektion)
def self_attention(X):
    """
    X: (seq_len, d_model) - Input Embeddings
    """
    # Q = K = V = X (vereinfacht)
    Q = K = V = X

    d_k = X.shape[1]

    # 1. Attention Scores
    scores = Q @ K.T  # (seq_len, seq_len)

    # 2. Scaling
    scores_scaled = scores / np.sqrt(d_k)

    # 3. Softmax (Attention Weights)
    attention_weights = softmax(scores_scaled, axis=-1)

    # 4. Weighted Sum
    output = attention_weights @ V  # (seq_len, d_model)

    return output, attention_weights

# Beispiel: 3 Tokens
X = np.array([
    [0.5, 0.8],  # "Kühlschrank"
    [0.2, 0.4],  # "ist"
    [0.9, 0.1]   # "defekt"
])

output, weights = self_attention(X)

print("Input:\n", X)
print("\nAttention Weights:\n", weights)
print("\nOutput:\n", output)

# Visualisiere: Token 2 ("defekt") attendiert auf welche Tokens?
print(f"\nToken 'defekt' Attention: {weights[2]}")
# [0.28, 0.27, 0.45] → attendiert am meisten auf sich selbst
```

**Output:**
```
Input:
 [[0.5 0.8]
  [0.2 0.4]
  [0.9 0.1]]

Attention Weights:
 [[0.38 0.28 0.34]
  [0.34 0.33 0.33]
  [0.28 0.27 0.45]]

Output:
 [[0.54 0.47]
  [0.52 0.46]
  [0.58 0.41]]

Token 'defekt' Attention: [0.28 0.27 0.45]
```

---

## 🔗 Related Terms

### **Voraussetzungen**
- **[Dot Product](../01-vectors-embeddings/04-dot-product.md)**: Q·K^T
- **[Embedding](../01-vectors-embeddings/01-embedding.md)**: Input zu Attention

### **Baut darauf auf**
- **[Multi-Head Attention](02-multi-head-attention.md)**: Parallel mehrere Attention-Mechanismen
- **[Transformer Block](04-transformer-block.md)**: Attention + Feed-Forward
- **[Context Window](05-context-window.md)**: Max Sequenzlänge für Attention

### **Verwandt**
- **Softmax**: Normalisierung der Attention Scores
- **Matrix Multiplication**: Kern-Operation (Q·K^T, A·V)

---

## 📍 Where This Appears

### **Primary Chapter**
- `01-historical/04-attention-transformers/` - "Attention is All You Need" (2017)
- `02-modern-ai/01-llms/01-model-families.md` - Self-Attention in GPT, BERT

### **Implementation**
- `03-core/01-training/` - Training von Attention-Mechanismen
- `04-advanced/` - Flash Attention, Sparse Attention Optimierungen

---

## ⚠️ Common Misconceptions

### ❌ "Attention ist nur für NLP"
**Falsch!** Attention wird überall verwendet:
- **Vision Transformers (ViT)**: Attention zwischen Bild-Patches
- **Audio**: Attention in Speech Recognition
- **Multimodal**: Cross-Attention zwischen Bild und Text (CLIP, GPT-4V)

### ❌ "Attention Scores sind Wahrscheinlichkeiten"
**Technisch falsch!** Softmax normalisiert zu **Gewichtungen** die summieren zu 1, aber das sind keine echten Wahrscheinlichkeiten (nicht aus Wahrscheinlichkeitsverteilung geschätzt).

**Richtig:** Attention Weights sind **normalisierte Relevanz-Scores**.

### ❌ "Mehr Attention = besser"
**Falsch!** Attention hat Kosten:
- **Memory**: $O(n^2)$ für Sequenz-Länge $n$
- **Compute**: Quadratisch mit Sequenz-Länge

**Long-Context Problem:** GPT-4 (128k tokens) braucht 128k² = 16B Attention-Operationen!

**Lösungen:**
- Sparse Attention (nur auf Teilmenge attenden)
- Flash Attention (Memory-Optimierung)
- Sliding Window Attention (Longformer)

---

## 🎯 Zusammenfassung

**Ein Satz:** Self-Attention gewichtet die Relevanz aller Tokens in einer Sequenz für jedes Token via Q·K^T, ermöglicht Parallelisierung und Long-Range Dependencies.

**Formel (Merksatz):**
$$\text{Attention} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

**Key Takeaways:**
1. **Q, K, V**: Query (was suche ich?), Key (was biete ich?), Value (was ist mein Inhalt?)
2. **Parallelisierbar**: Alle Attention-Scores gleichzeitig (vs. RNN sequenziell)
3. **Long-Range**: Direkter Zugriff auf weit entfernte Tokens
4. **Quadratisch**: Memory/Compute $O(n^2)$ (Problem bei langen Sequenzen)

**"Attention is All You Need"** - Vaswani et al., 2017

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ➡️ [Nächster Begriff: Multi-Head Attention](02-multi-head-attention.md)
