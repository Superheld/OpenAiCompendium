# Quantization & Optimization: Glossar

## 🎯 Übersicht

Diese Kategorie definiert **Modelloptimierung und Ressourcen-Effizienz** - wie man große LLMs auf Consumer-Hardware lauffähig macht.

**Kernfrage:** Wie macht man 70B Parameter Modelle auf 24GB VRAM lauffähig?

---

## 📋 Begriffe in dieser Kategorie

### **Kern-Konzept**
1. **[01-quantization.md](01-quantization.md)** - Quantization / Model Quantization
   - **KRITISCH**: Von 16GB → 4GB ohne großen Qualitätsverlust
   - FP16 → INT8 → INT4 Kompression
   - Trade-off: Memory vs. Accuracy

---

## 🔗 Lernpfad

```
1. Quantization (01) → verstehe das Kern-Konzept
```

---

## 🎓 Was du danach kannst

- ✅ **Berechnen** Memory-Footprint eines Modells
- ✅ **Entscheiden** welche Quantization-Methode (GGUF vs. GPTQ vs. AWQ)
- ✅ **Abschätzen** Qualitätsverlust bei verschiedenen Bit-Tiefen

---

**Navigation:**
- 🏠 [Zurück zum Glossar](../00-overview.md)
- ⬅️ [Vorige Kategorie: Transformers & Attention](../02-transformers-attention/)
- ➡️ [Nächste Kategorie: RAG Concepts](../04-rag-concepts/)
