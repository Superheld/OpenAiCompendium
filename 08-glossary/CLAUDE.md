# CLAUDE.md - Glossar Template & Guidelines

Dieses Dokument definiert **Template und Qualitätsstandards** für Glossar-Einträge im ML Kompendium.

---

## 🎯 Purpose: Single Source of Truth

Das Glossar ist der **zentrale Referenzpunkt** für alle technischen Begriffe. Es löst drei Probleme:

1. **Redundanz-Vermeidung**: Begriffe wie "Quantization" (224 Zeilen!), "Token" (18 Dateien), "Chunking" (12 Dateien) werden nur einmal erklärt
2. **Konsistenz**: Mathematische Notation, Code-Beispiele, Definitionen standardisiert
3. **Wartbarkeit**: Update an einer Stelle → automatisch überall aktuell

---

## 📋 Wann gehört ein Begriff ins Glossar?

### **JA, ins Glossar wenn:**
✅ Begriff erscheint in **3+ Kapiteln**
✅ Begriff hat **unterschiedliche Definitionen** in verschiedenen Kapiteln
✅ Begriff ist **fundamental** für Verständnis (z.B. Tensor, Token, Embedding)
✅ Begriff hat **viele Aliase** (z.B. "Dense Vector" = "Embedding" = "Vector Representation")
✅ Begriff **nimmt viel Platz** in Kapiteln ein (>100 Zeilen Erklärung)

### **NEIN, nicht ins Glossar wenn:**
❌ Begriff ist **kapitel-spezifisch** (nur in einem Kontext relevant)
❌ Begriff ist **selbsterklärend** (keine Erklärung nötig)
❌ Begriff ist **zu nischig** (nur in einem Paper erwähnt)
❌ Begriff ist **kurz erklärbar** (<50 Worte) → kann inline bleiben

---

## 📐 Template-Struktur

Jeder Glossar-Eintrag folgt diesem Template:

```markdown
# [Begriff] / [Alias 1] / [Alias 2]

## Quick Definition

[1-2 Sätze: Was ist es? Wofür wird es genutzt?]

**Kategorie:** [Vectors/Transformers/RAG/etc.]
**Schwierigkeit:** [Beginner/Intermediate/Advanced]
**Aliases:** [Alle Varianten des Namens]

---

## 🧠 Detaillierte Erklärung

### Intuitive Erklärung

[Alltagsanalogie OHNE Formeln. "Stell dir vor..."]

### Mathematische Formalisierung

[Formeln MIT Intuition. Schritt-für-Schritt Ableitungen.]

$$\text{Formel mit Bedeutung}$$

**Komponenten:**
- $x$ = ...
- $y$ = ...

### Why It Matters - Welches Problem löst [Begriff]?

**Problem 1: [Name]** ❌ → ✅
[Beschreibung des Problems OHNE den Begriff]
[Beschreibung der Lösung MIT dem Begriff]

**Problem 2: [Name]** ❌ → ✅
[...]

**Problem 3: [Name]** ❌ → ✅
[...]

### Common Variations

[Verschiedene Implementierungen/Varianten des Konzepts]

---

## 💻 Code-Beispiel

```python
# Kommentierter, lauffähiger Code
# MUSS copy-paste-ready sein!

# Ausgabe zeigen
```

**Output:**
```
[Erwartete Ausgabe]
```

---

## 🔗 Related Terms

### **Voraussetzungen** (was du vorher wissen solltest)
- **[Begriff](link.md)**: Kurze Erklärung warum relevant

### **Baut darauf auf** (was danach kommt)
- **[Begriff](link.md)**: Kurze Erklärung warum relevant

### **Verwandt**
- **[Begriff](link.md)**: Kurze Erklärung der Verbindung

---

## 📍 Where This Appears

### **Primary Chapter** (detaillierte Erklärung)
- `pfad/zu/kapitel.md` (Sektion X) - Beschreibung

### **Usage Examples** (praktische Anwendung)
- `pfad/zu/anwendung.md` - Wie verwendet

### **Mentioned In** (weitere Vorkommen)
- `pfad/zu/erwähnung.md` - Kontext

---

## ⚠️ Common Misconceptions

### ❌ "Misconception 1"
**Falsch!** [Warum falsch]

[Beispiel das Misconception zeigt]

**Richtig:** [Korrekte Erklärung]

### ❌ "Misconception 2"
[...]

### ❌ "Misconception 3"
[...]

---

## 🎯 Zusammenfassung

**Ein Satz:** [Kernaussage in einem Satz]

**Formel (Merksatz):**
$$\text{Hauptformel}$$

**Key Takeaways:**
1. [Wichtigster Punkt]
2. [Zweitwichtigster Punkt]
3. [Drittwichtigster Punkt]
4. [Viertwich...]

**Wann nutzen?**
- ✅ [Use Case 1]
- ✅ [Use Case 2]
- ✅ [Use Case 3]
- ❌ [Anti-Pattern 1]

---

**Navigation:**
- 🏠 [Zurück zur Kategorie](00-overview.md)
- ⬅️ [Vorheriger Begriff: X](XX-prev.md)
- ➡️ [Nächster Begriff: Y](YY-next.md)
```

---

## 🎯 Qualitätsstandards

Jeder Glossar-Eintrag MUSS haben:

### **Pflicht-Komponenten (100% Coverage):**
- ✅ **Quick Definition** (1-2 Sätze)
- ✅ **Kategorie & Schwierigkeit**
- ✅ **Aliases** (alle Varianten)
- ✅ **Intuitive Erklärung** (ohne Formeln)
- ✅ **"Why It Matters"** (welches Problem löst es?)
- ✅ **Code-Beispiel** (lauffähig!)
- ✅ **Related Terms** (Cross-References)
- ✅ **Common Misconceptions** (mind. 3)
- ✅ **Zusammenfassung** (Ein-Satz + Key Takeaways)

### **Optional aber empfohlen:**
- ⭐ **Mathematische Formalisierung** (bei technischen Begriffen)
- ⭐ **Vergleichstabellen** (wenn mehrere Varianten existieren)
- ⭐ **Benchmarks/Zahlen** (Performance-Daten)
- ⭐ **Trade-off Analysis** (Vor- und Nachteile)

---

## 📊 Beispiel-Entries als Vorbilder

### **Vorbild: Tensor** (`01-vectors-embeddings/00-tensor.md`)
✅ Erklärt **4 Probleme** die Tensoren lösen
✅ GPU-Benchmark (79× Speedup)
✅ Automatic Differentiation Code-Beispiel
✅ Hierarchie (0D→1D→2D→3D→4D)
✅ 3 Common Misconceptions

### **Vorbild: Quantization** (`03-quantization-optimization/01-quantization.md`)
✅ **3 Probleme** (Memory, Speed, Cost) mit Zahlen
✅ Memory-Berechnung Formeln
✅ Vergleichstabelle (GGUF vs GPTQ vs AWQ vs bitsandbytes)
✅ QLoRA Fine-Tuning Code
✅ Benchmark-Daten (Perplexity-Vergleich)

### **Vorbild: Chunking** (`04-rag-concepts/02-chunking.md`)
✅ Kompakt (für kompaktere Begriffe)
✅ **Decision Table** (Chunk Size Trade-offs)
✅ 3 Chunking-Strategien verglichen
✅ Token-aware Code-Beispiel

---

## 🔗 Cross-Reference Guidelines

### **Linking-Format:**
```markdown
**[Begriff-Name](../kategorie/XX-begriff.md)**: Kurze Erklärung
```

### **Wann verlinken?**
- **Voraussetzungen**: Begriff wird vorher benötigt
- **Baut darauf auf**: Begriff nutzt aktuellen Begriff
- **Verwandt**: Ähnliches Konzept oder Alternative

### **Beispiel:**
```markdown
### **Voraussetzungen**
- **[Tensor](../01-vectors-embeddings/00-tensor.md)**: Embeddings sind Tensoren
- **[Token](../05-llm-training/01-token.md)**: Chunk-Size in Tokens gemessen

### **Baut darauf auf**
- **[RAG](01-rag.md)**: Nutzt Chunks für Retrieval

### **Verwandt**
- **[Context Window](../02-transformers-attention/05-context-window.md)**: Bestimmt optimale Chunk-Size
```

---

## 📝 Schreibstil-Guidelines

### **Sprache:**
- **Deutsch** als Hauptsprache
- **Technische Präzision** ohne Jargon
- **Du-Form** für direkte Ansprache ("du kannst...", "verstehe...")

### **Ton:**
- **Educational**: Erkläre, lehre nicht nur dokumentiere
- **Problem-First**: Zeige WARUM der Begriff wichtig ist
- **Honest**: Limitations und Trade-offs explizit nennen

### **Formatierung:**
- **Emojis** für Navigation (🎯 🧠 💻 🔗 ⚠️ 🎯)
- **Code-Blocks** mit Syntax-Highlighting
- **Tabellen** für Vergleiche
- **Formeln** in LaTeX ($...$ inline, $$...$$ block)

### **Code-Beispiele:**
- **Lauffähig**: Muss copy-paste funktionieren
- **Kommentiert**: Auf Deutsch
- **Minimal**: Nur das Nötigste
- **Output**: Zeige erwartete Ausgabe

---

## 🚀 Workflow: Neuen Begriff hinzufügen

### **1. Check ob Glossar-würdig**
- [ ] Erscheint in 3+ Kapiteln?
- [ ] Fundamental für Verständnis?
- [ ] Viele Aliase oder Varianten?

### **2. Kategorie bestimmen**
```
01-vectors-embeddings/     → Vektoren, Embeddings, Retrieval
02-transformers-attention/ → Transformer, Attention, Context
03-quantization-optimization/ → Quantization, LoRA, MoE
04-rag-concepts/           → RAG, Chunking, Re-Ranking
05-llm-training/           → Token, Fine-Tuning, Hallucination
06-evaluation-metrics/     → Metriken (Precision, NDCG, etc.)
```

### **3. Dateiname festlegen**
```
XX-begriff-name.md

XX = Nummer (01, 02, 03, ...)
begriff-name = Kleinbuchstaben, Bindestriche

Beispiel: 01-embedding.md, 02-cosine-similarity.md
```

### **4. Template ausfüllen**
- Kopiere Template von oben
- Fülle alle Pflicht-Komponenten aus
- Schreibe lauffähigen Code
- Identifiziere 3+ Common Misconceptions
- Füge Cross-References hinzu

### **5. Overview-Datei aktualisieren**
In `XX-kategorie/00-overview.md`:
```markdown
X. **[XX-begriff.md](XX-begriff.md)** - Begriff / Alias
   - Quick description
   - Warum wichtig
```

### **6. Haupt-Overview aktualisieren**
In `/00-overview.md` Glossar-Sektion:
- Anzahl der Begriffe updaten
- Neue Kategorie ergänzen (falls neu)

---

## 📊 Qualitäts-Checkliste

Vor Commit, prüfe:

- [ ] **Quick Definition** vorhanden (1-2 Sätze)
- [ ] **Aliases** aufgelistet
- [ ] **"Why It Matters"** mit mind. 1 Problem
- [ ] **Code-Beispiel** lauffähig (getestet!)
- [ ] **3+ Common Misconceptions**
- [ ] **Related Terms** mit Links
- [ ] **Zusammenfassung** mit "Ein Satz" + Key Takeaways
- [ ] **Navigation** (Zurück, Vorheriger, Nächster)
- [ ] **Deutsche Sprache** durchgängig
- [ ] **Formeln** mit Intuition erklärt
- [ ] **Problem-First** Approach (nicht nur Definition!)

---

## 🎯 Anti-Patterns (Was NICHT tun)

❌ **Nur Definition ohne "Why It Matters"**
```markdown
# Begriff
Begriff ist X.
```
→ Erkläre WARUM der Begriff existiert, welches Problem er löst!

❌ **Code ohne Output**
```python
x = model.encode("text")
```
→ Zeige erwartete Ausgabe!

❌ **Formel ohne Intuition**
```markdown
$$\text{formula}$$
```
→ Erkläre WAS die Formel bedeutet, nicht nur zeigen!

❌ **Keine Cross-References**
→ Jeder Begriff existiert in Relation zu anderen!

❌ **Misconceptions fehlen**
→ Was glauben Leute fälschlicherweise? Korrigiere explizit!

❌ **Zu akademisch**
```markdown
"Der Begriff X bezeichnet die Menge aller..."
```
→ Nutze "du"-Form und Alltagsanalogie!

---

## 📚 Ressourcen

**Template-Datei:**
- Siehe oben unter "Template-Struktur"

**Beispiel-Einträge:**
- `01-vectors-embeddings/00-tensor.md` (umfangreich)
- `03-quantization-optimization/01-quantization.md` (mit Benchmarks)
- `04-rag-concepts/02-chunking.md` (kompakt)

**Cross-Reference Guide:**
- Siehe "Cross-Reference Guidelines" oben

---

**Happy Glossary Writing! 📚✨**

*Erstellt: 2025-10-26*
*Letzte Aktualisierung: 2025-10-26*
