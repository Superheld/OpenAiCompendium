# Tensor / Mehrdimensionales Array

## Quick Definition

Die fundamentale Datenstruktur in Deep Learning - ein n-dimensionales Array von Zahlen, das auf GPUs parallelisiert werden kann und automatisches Differentiation ermöglicht.

**Kategorie:** Vectors & Embeddings (Fundamentals)
**Schwierigkeit:** Beginner (Konzept), Intermediate (Operations)
**Aliases:** Tensor, Mehrdimensionales Array, n-dimensional Array

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

Ein **Tensor** ist wie ein "Container" für Zahlen, der verschiedene Formen haben kann:

**Analogie: Excel-Tabellen**
- **Skalar** (0D Tensor) = eine einzelne Zelle: `42`
- **Vektor** (1D Tensor) = eine Spalte: `[1, 2, 3, 4]`
- **Matrix** (2D Tensor) = eine Tabelle: `[[1, 2], [3, 4]]`
- **3D Tensor** = ein Stapel von Tabellen (z.B. RGB-Bild: Höhe × Breite × 3 Farbkanäle)
- **4D Tensor** = ein Batch von Bildern (Batch × Höhe × Breite × Kanäle)

**Wichtig:** Tensoren sind nicht nur Arrays - sie sind **GPU-beschleunigt** und **differenzierbar** (für Backpropagation)!

### Mathematische Formalisierung

**Tensor-Hierarchie:**

| Dimension | Name | Mathematik | Beispiel | Shape |
|-----------|------|------------|----------|-------|
| **0D** | Skalar | $x \in \mathbb{R}$ | `5.0` | `()` |
| **1D** | Vektor | $\mathbf{v} \in \mathbb{R}^n$ | `[1, 2, 3]` | `(3,)` |
| **2D** | Matrix | $\mathbf{M} \in \mathbb{R}^{m \times n}$ | `[[1, 2], [3, 4]]` | `(2, 2)` |
| **3D** | 3D-Tensor | $\mathbf{T} \in \mathbb{R}^{d_1 \times d_2 \times d_3}$ | RGB-Bild | `(224, 224, 3)` |
| **4D+** | Batch | $\mathbf{B} \in \mathbb{R}^{b \times d_1 \times \ldots}$ | Batch von Bildern | `(32, 224, 224, 3)` |

**Tensor-Eigenschaften:**

1. **Shape** (Form): Dimensionen des Tensors
   $$\text{shape}(\mathbf{T}) = (d_1, d_2, \ldots, d_n)$$

2. **Rank** (Rang): Anzahl der Dimensionen
   $$\text{rank}(\mathbf{T}) = n$$

3. **Size** (Größe): Gesamtzahl der Elemente
   $$\text{size}(\mathbf{T}) = d_1 \times d_2 \times \cdots \times d_n$$

**Beispiel:**
```
Tensor Shape: (32, 384)
- Rank: 2 (Matrix)
- Dimension 0: 32 (Batch Size)
- Dimension 1: 384 (Embedding Dimension)
- Size: 32 × 384 = 12,288 Elemente
```

### Why It Matters - Welches Problem lösen Tensoren?

**Problem 1: Batch-Verarbeitung ohne Tensoren** ❌

**Vor Tensoren (naive Python):**
```python
# Verarbeite 1000 Bilder einzeln (langsam!)
results = []
for image in images:  # 1000 Iterationen
    result = model.process(image)  # CPU, sequenziell
    results.append(result)
# Dauert: 100 Sekunden (0.1s pro Bild)
```

**Mit Tensoren:** ✅
```python
# Verarbeite alle 1000 Bilder PARALLEL auf GPU!
images_tensor = torch.stack(images)  # Shape: (1000, 224, 224, 3)
results = model(images_tensor)  # GPU, parallel
# Dauert: 0.5 Sekunden (200× schneller!)
```

**Problem 2: Automatisches Differentiation** ❌ → ✅

**Ohne Tensoren:**
```python
# Gradient von Hand berechnen (fehleranfällig!)
def backward_pass(x, y, loss):
    # Manuell für jede Operation die Ableitung programmieren
    grad_x = ... # Komplexe Kettenregel
    grad_y = ... # Fehleranfällig!
    return grad_x, grad_y
```

**Mit Tensoren (PyTorch):**
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
loss = (x ** 2).sum()
loss.backward()  # Automatisch alle Gradienten berechnet!
print(x.grad)  # [2.0, 4.0] ← Gradienten automatisch!
```

**Problem 3: GPU-Beschleunigung** ❌ → ✅

**Ohne Tensoren (NumPy auf CPU):**
```python
import numpy as np
a = np.random.rand(10000, 10000)
b = np.random.rand(10000, 10000)
c = a @ b  # CPU: 15 Sekunden
```

**Mit Tensoren (PyTorch auf GPU):**
```python
import torch
a = torch.rand(10000, 10000, device='cuda')
b = torch.rand(10000, 10000, device='cuda')
c = a @ b  # GPU: 0.05 Sekunden (300× schneller!)
```

**Problem 4: Memory-Effizienz bei Deep Learning**

**Warum nicht Python Lists?**
```python
# Python List: Jedes Element ist ein Python-Objekt (overhead!)
python_list = [1.0, 2.0, 3.0]
# Memory: 3 × 28 bytes = 84 bytes (Python Object overhead)

# Tensor: Kontinuierlicher Speicher
tensor = torch.tensor([1.0, 2.0, 3.0])
# Memory: 3 × 4 bytes = 12 bytes (nur Daten, kein overhead)
```

**Bei 1 Million Zahlen:**
- Python List: 28 MB
- Tensor: 4 MB (7× weniger!)

### Common Variations

**1. PyTorch Tensor** (Standard in Deep Learning)
```python
import torch
x = torch.tensor([1, 2, 3])
x = x.to('cuda')  # GPU
x.requires_grad = True  # Für Training
```

**2. NumPy Array** (CPU-only, kein Autograd)
```python
import numpy as np
x = np.array([1, 2, 3])
# Kein GPU, keine automatische Differentiation
```

**3. TensorFlow Tensor**
```python
import tensorflow as tf
x = tf.constant([1, 2, 3])
```

**4. JAX Array** (für High-Performance Computing)
```python
import jax.numpy as jnp
x = jnp.array([1, 2, 3])
```

---

## 💻 Code-Beispiel: Tensoren in Action

```python
import torch
import numpy as np

# ========================================
# 1. TENSOR-HIERARCHIE
# ========================================

# 0D Tensor (Skalar)
scalar = torch.tensor(42.0)
print(f"Skalar: {scalar}, Shape: {scalar.shape}, Rank: {scalar.ndim}")
# Output: Skalar: 42.0, Shape: torch.Size([]), Rank: 0

# 1D Tensor (Vektor)
vector = torch.tensor([1.0, 2.0, 3.0])
print(f"Vektor: {vector}, Shape: {vector.shape}, Rank: {vector.ndim}")
# Output: Vektor: [1, 2, 3], Shape: torch.Size([3]), Rank: 1

# 2D Tensor (Matrix)
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Matrix:\n{matrix}\nShape: {matrix.shape}, Rank: {matrix.ndim}")
# Output: Shape: torch.Size([2, 2]), Rank: 2

# 3D Tensor (z.B. RGB-Bild: H×W×C)
image = torch.rand(224, 224, 3)  # Höhe × Breite × Kanäle
print(f"Bild Shape: {image.shape}, Rank: {image.ndim}")
# Output: Bild Shape: torch.Size([224, 224, 3]), Rank: 3

# 4D Tensor (Batch von Bildern: B×H×W×C)
batch = torch.rand(32, 224, 224, 3)  # 32 Bilder
print(f"Batch Shape: {batch.shape}, Rank: {batch.ndim}")
# Output: Batch Shape: torch.Size([32, 224, 224, 3]), Rank: 4

# ========================================
# 2. WARUM TENSOREN? - GPU-BESCHLEUNIGUNG
# ========================================

# CPU (NumPy)
import time
a_np = np.random.rand(5000, 5000)
b_np = np.random.rand(5000, 5000)

start = time.time()
c_np = a_np @ b_np
cpu_time = time.time() - start
print(f"\nCPU (NumPy): {cpu_time:.3f}s")

# GPU (PyTorch) - wenn verfügbar
if torch.cuda.is_available():
    a_gpu = torch.rand(5000, 5000, device='cuda')
    b_gpu = torch.rand(5000, 5000, device='cuda')

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    print(f"GPU (PyTorch): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time/gpu_time:.1f}×")
    # Typisch: 50-100× schneller!

# ========================================
# 3. AUTOMATISCHES DIFFERENTIATION
# ========================================

# Berechne Gradienten automatisch!
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Forward Pass
y = x ** 2  # [4.0, 9.0]
loss = y.sum()  # 13.0

# Backward Pass (automatisch!)
loss.backward()

print(f"\nInput: {x.data}")
print(f"Output: {y.data}")
print(f"Gradient: {x.grad}")  # [4.0, 6.0] = d(x²)/dx = 2x
# Gradient automatisch berechnet - keine manuelle Ableitung!

# ========================================
# 4. TENSOR-OPERATIONEN (wie in LLMs)
# ========================================

# Embedding-Lookup (wie in GPT)
vocab_size, embed_dim = 50000, 768
embedding_table = torch.randn(vocab_size, embed_dim)

# Token IDs → Embeddings
token_ids = torch.tensor([42, 123, 9999])  # 3 Tokens
embeddings = embedding_table[token_ids]  # Shape: (3, 768)
print(f"\nToken Embeddings Shape: {embeddings.shape}")

# Batch Matrix Multiplication (wie in Attention)
batch_size, seq_len, d_model = 32, 512, 768
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)

# Attention Scores: Q @ K^T
attention_scores = torch.matmul(Q, K.transpose(-2, -1))
print(f"Attention Scores Shape: {attention_scores.shape}")
# Shape: (32, 512, 512) - jedes Token vs. alle Tokens
```

**Output:**
```
Skalar: 42.0, Shape: torch.Size([]), Rank: 0
Vektor: [1, 2, 3], Shape: torch.Size([3]), Rank: 1
Matrix:
[[1. 2.]
 [3. 4.]]
Shape: torch.Size([2, 2]), Rank: 2
Bild Shape: torch.Size([224, 224, 3]), Rank: 3
Batch Shape: torch.Size([32, 224, 224, 3]), Rank: 4

CPU (NumPy): 2.456s
GPU (PyTorch): 0.031s
Speedup: 79.2×

Input: tensor([2., 3.])
Output: tensor([4., 9.])
Gradient: tensor([4., 6.])

Token Embeddings Shape: torch.Size([3, 768])
Attention Scores Shape: torch.Size([32, 512, 512])
```

---

## 🔗 Related Terms

### **Spezialfälle von Tensoren**
- **[Embedding](01-embedding.md)**: 1D Tensor (Vektor) oder 2D Tensor (Batch von Vektoren)
- **[Dot Product](04-dot-product.md)**: Operation zwischen 1D Tensoren
- **Matrix Multiplication**: Operation zwischen 2D Tensoren

### **Nutzt Tensoren**
- **[Self-Attention](../02-transformers-attention/01-self-attention.md)**: Q, K, V sind Tensoren
- **Neural Networks**: Alle Gewichte und Aktivierungen sind Tensoren
- **Backpropagation**: Gradient-Tensoren

### **Verwandt**
- **NumPy Array**: CPU-only Version (kein Autograd)
- **GPU Computing**: Tensoren ermöglichen GPU-Beschleunigung
- **Automatic Differentiation**: Gradienten automatisch aus Tensor-Operationen

---

## 📍 Where This Appears

### **Fundamental für:**
- **ALLE** Deep Learning Modelle (GPT, BERT, CNNs, RNNs)
- **Embeddings**: Tensoren mit Shape (batch, embedding_dim)
- **Attention**: Q, K, V Tensoren mit Shape (batch, seq_len, d_model)
- **Training**: Gewichte, Gradienten - alles Tensoren

### **Frameworks:**
- **PyTorch**: `torch.Tensor`
- **TensorFlow**: `tf.Tensor`
- **JAX**: `jax.numpy.array`

### **Primary Chapters:**
- `03-core/01-training/` - Tensor-Operationen im Training
- `02-modern-ai/01-llms/` - Tensoren in LLM-Architekturen

---

## ⚠️ Common Misconceptions

### ❌ "Tensoren sind nur mehrdimensionale Arrays"
**Unvollständig!** NumPy Arrays sind mehrdimensional, aber:

**NumPy Array:**
- ✅ Mehrdimensional
- ❌ Kein Autograd
- ❌ Kein GPU
- ❌ Nicht für Deep Learning optimiert

**Tensor (PyTorch/TensorFlow):**
- ✅ Mehrdimensional
- ✅ **Automatic Differentiation** (requires_grad=True)
- ✅ **GPU-beschleunigt** (.to('cuda'))
- ✅ **Deep Learning Primitives** (conv2d, attention, etc.)

**Richtig:** Tensoren = Arrays + Autograd + GPU + DL-Optimierungen

### ❌ "Mehr Dimensionen = komplexer"
**Falsch!** Dimensionen haben klare Bedeutung:

**4D Tensor: (32, 224, 224, 3)**
- Dimension 0: Batch (32 Bilder)
- Dimension 1: Höhe (224 Pixel)
- Dimension 2: Breite (224 Pixel)
- Dimension 3: Kanäle (RGB = 3)

**Nicht "4D-Raum"** - sondern strukturierte Daten!

**Merksatz:** Jede Dimension hat praktische Bedeutung (Batch, Sequenz, Features, etc.)

### ❌ "Tensoren sind langsam wegen vielen Dimensionen"
**Genau umgekehrt!** Tensoren sind schnell **wegen** Batch-Dimensionen:

**Ohne Batch (langsam):**
```python
for i in range(1000):
    output = model(data[i])  # 1000× Overhead
# 10 Sekunden
```

**Mit Batch (schnell):**
```python
output = model(data)  # Shape: (1000, ...), parallel!
# 0.1 Sekunden (100× schneller!)
```

**GPU-Parallelismus** funktioniert nur mit Batch-Dimensionen!

### ❌ "Tensor Shape ist egal, wird automatisch gehandhabt"
**SEHR FALSCH!** Shape Mismatches sind der häufigste Fehler:

```python
# ERROR!
x = torch.rand(32, 10)  # (batch, features)
y = torch.rand(10, 5)   # (features, outputs)
z = x + y  # RuntimeError: Shape mismatch!

# Korrekt:
z = x @ y  # Matrix multiplication: (32, 10) @ (10, 5) = (32, 5)
```

**Best Practice:** Immer Shape kommentieren!
```python
x = model(input)  # Shape: (batch, seq_len, hidden_dim)
```

---

## 🎯 Zusammenfassung

**Ein Satz:** Tensoren sind GPU-beschleunigte, differenzierbare n-dimensionale Arrays - die fundamentale Datenstruktur für alle Deep Learning Operationen.

**3 Haupt-Vorteile:**
1. **GPU-Parallelisierung** → 100× schneller als CPU
2. **Automatic Differentiation** → Gradienten automatisch
3. **Batch-Verarbeitung** → Tausende Samples parallel

**Tensor-Hierarchie (Merksatz):**
```
0D (Skalar) → 1D (Vektor) → 2D (Matrix) → 3D (Bild) → 4D (Batch)
```

**Key Takeaways:**
- **Nicht nur Arrays**: Autograd + GPU + DL-Primitives
- **Shape matters**: Jede Dimension hat Bedeutung (Batch, Seq, Features)
- **GPU = Speed**: 50-100× schneller als CPU NumPy
- **Backprop**: Gradienten automatisch aus Tensor-Operationen

**Warum gibt es Tensoren?**
→ Ohne Tensoren: Kein GPU-Training, keine automatische Differentiation, kein modernes Deep Learning! 🚀

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ➡️ [Nächster Begriff: Embedding](01-embedding.md)
