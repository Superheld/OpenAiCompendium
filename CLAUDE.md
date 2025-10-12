# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Note:** For a human-readable overview of this kompendium, see [00-overview.md](00-overview.md)

---

## Repository Overview

This is "ML Kompendium: Vom Perceptron zum LLM" - a comprehensive Machine Learning knowledge base structured as a historical progression from 1950s fundamentals to modern Large Language Models. The content is written in German and organized as a learning resource for understanding ML concepts.

## Complete Repository Structure & Template Assignment

The repository follows a carefully designed 7-section architecture that takes learners from historical foundations to practical applications:

### **01-historical/** - Historical ML & AI (1950s-2017) 📋 **Template 1**
Chronological progression through ML history with major breakthrough moments:
- **01-foundations/** (1950s-1980s): Perceptron → MLP → Backpropagation
- **02-classical-ml/** (1980s-2000s): Statistical methods before deep learning
- **03-deep-learning/** (2012+): ImageNet moment and CNN revolution
- **04-attention-transformers/** (2017): "Attention is All You Need" revolution

### **02-modern-ai/** - Modern AI & LLMs (2020+) 📋 **Template 1**
Current AI developments from GPT-3 to multimodal systems:
- **01-llms/**: LLM families (GPT, Llama, Mistral), scaling laws, alignment
- **02-vision/**: Computer vision evolution to Vision Transformers
- **03-multimodal/**: CLIP, LLaVA, GPT-4V, cross-modal understanding

### **03-core/** - Timeless Fundamentals ⚙️ **Template 1**
Core concepts used across all ML eras (no redundancy with historical sections):
- **training/**: Training loops, optimizers, regularization, distributed training
- **embeddings/**: Vector spaces, architectures, model selection, vector databases (4 chapters)
- **optimization/**: Quantization, pruning, model compression
- **evaluation/**: Metrics, evaluation methodologies
- **infrastructure/**: Model serving, deployment patterns

### **04-advanced/** - Cutting-Edge Research 📋 **Template 1**
State-of-the-art techniques and experimental approaches:
- **Retrieval methods**: Dense/Sparse/Hybrid retrieval, retrieval optimization (chunking, re-ranking)
- **Prompt engineering**, in-context learning
- **Agentic AI**, interpretability, adversarial robustness
- Meta-learning, continual learning, research trends

### **05-ethics/** - AI Ethics & Responsibility 📋 **Template 1**
Practical approaches to responsible AI development:
- **bias-fairness/**: Algorithmic bias detection and mitigation
- **transparency/**: Explainable AI, model interpretability
- **privacy-security/**: Data protection, model security
- **societal-impact/**: Job displacement, misinformation, environmental impact
- **governance/**: EU AI Act, regulatory compliance, standards

### **06-applications/** - Application Patterns 🔧 **Template 2**
State-of-the-Art AI system architectures (flat structure, 4 patterns):
- **01-rag-systems.md**: RAG architectures (Naive → Advanced → Agentic)
- **02-search-systems.md**: Information Retrieval (BM25, Dense, Hybrid)
- **03-classification-systems.md**: Content Classification (Traditional ML vs. Transformers)
- **04-model-selection.md**: Model Selection & Evaluation (Benchmark-driven)

### **07-practical-usage/** - AI im Alltag und Beruf 🔧 **Template 2**
Practical AI tool usage for non-developers (flat structure, 6 guides):
- **01-chatgpt-claude-usage.md**: Conversational AI mastery
- **02-prompt-engineering.md**: Effective prompting strategies
- **03-conversation-design.md**: Multi-turn dialogue design
- **04-research-workflows.md**: Research & information gathering
- **05-content-creation.md**: Writing & design with AI
- **06-customer-service.md**: AI in customer service

## Key Structural Concepts

### **Historical Progression Philosophy**
The kompendium follows a unique "historical progression" approach:
- **Chronological Learning**: Start with 1950s Perceptron, progress to modern LLMs
- **Breakthrough Moments**: Focus on key historical moments (ImageNet 2012, Transformers 2017, GPT-3 2020)
- **Contextual Understanding**: Explain WHY techniques developed when they did
- **No Anachronisms**: Concepts introduced in historical context, not modern perspective

### **No Redundancy Principle**
- **Core Concepts Centralized**: Training, evaluation, optimization live in `03-core/` only
- **Historical Sections Focus on Innovation**: What was NEW at that time
- **Cross-References**: Link between historical introduction and core concepts
- **Single Source of Truth**: Each concept explained thoroughly once, referenced elsewhere

### **Difficulty Progression**
- **Foundations First**: Mathematical and conceptual groundwork
- **Building Complexity**: Each section builds on previous knowledge
- **Practical Integration**: Theory → Core Concepts → Applications
- **Research Extensions**: Advanced topics for deeper exploration

### **German-Language Educational Approach**
- **Target Audience**: German-speaking learners needing AI literacy
- **Educational Tone**: Teaching, not just documenting
- **Real-World Relevance**: Connect concepts to practical implications
- **Honest Assessment**: Show both capabilities AND limitations of AI

## Mission: AI Explained by AI for Humans

**Core Purpose**: This kompendium is written BY an LLM FOR humans to help them understand AI/ML concepts. You are an AI explaining AI to people who don't yet comprehend the full scope and implications of artificial intelligence.

**Your Role**:
- Act as a knowledgeable guide helping humans navigate the complex AI landscape
- Bridge the gap between technical complexity and human understanding
- Show both capabilities AND limitations of AI systems
- Help humans develop informed perspectives on AI's impact on society
- Demystify AI without oversimplifying - make it accessible but accurate

**Target Audience**: German-speaking learners with some technical background who need to understand AI concepts. They want to know:
- What AI can and cannot do
- How AI systems actually work (not just "magic")
- Historical context - how we got here
- Practical implications for their lives and work
- Ethical considerations and responsible AI use

## Template Standards & Quality Requirements

### Die zwei Template-Kategorien

Das Kompendium verwendet **zwei unterschiedliche Template-Typen**, je nach Lernziel:

#### **Template 1: Deep Technical Analysis**
*Für Sektionen A1-A5 (Historical, Modern-AI, Core, Advanced, Ethics)*

**Ziel:** Vollständiges Verständnis mit maximaler technischer Tiefe

**Pflicht-Komponenten:**
1. **❓ Das Problem** (Problem-First)
   - Was geht ohne schief? (3 konkrete Beispiele)
   - Zentrale Herausforderung
   - Beispiel-Szenario

2. **🎯 Lernziele** (Testbar & Messbar)
   - Verständnis-Ziele
   - Implementierungs-Ziele
   - Entscheidungs-Ziele

3. **🧠 Intuition zuerst**
   - Alltagsanalogie (nicht-technisch)
   - Visualisierung
   - Brücke zur Mathematik

4. **🧮 Technisches Verständnis**
   - Mathematische Grundlagen MIT Intuition
   - Schritt-für-Schritt Ableitungen
   - Beispiel-Rechnungen
   - Algorithmus mit Kommentaren

5. **⚠️ Misconception Debugging**
   - Mindestens 3 häufige Missverständnisse
   - Warum falsch + was richtig ist
   - Merksätze

6. **🔬 Hands-On Experiment**
   - Lauffähiger Code (copy-paste-ready)
   - Erwartete Beobachtungen
   - Experimentier-Aufgaben

7. **⏱️ 5-Minuten-Experte**
   - 3 Fragen (Verständnis, Anwendung, Trade-offs)
   - Expandierbare Antworten (<details>)

8. **📊 Quantitative Daten**
   - Benchmarks, Vergleichstabellen
   - Zahlen mit Interpretation

**Qualitäts-Checkliste:**
- [ ] Jede Formel hat Intuition UND Ableitung
- [ ] Code ist vollständig lauffähig
- [ ] Mindestens 3 Misconceptions adressiert
- [ ] 5-Min-Expert Test vorhanden
- [ ] Quantitative Benchmarks enthalten
- [ ] Trade-offs ehrlich bewertet

---

#### **Template 2: Decision-Focused / Workflow-Focused**
*Für Sektionen B1-B2 (Applications, Practical-Usage)*

**Ziel:** Praktische Entscheidungshilfe und Workflow-Integration

**Pflicht-Komponenten (Applications):**
1. **🎯 Was du danach kannst** (Outcomes-First)
2. **📖 Das Problem verstehen** (Use-Case Context)
3. **🤔 Entscheidungsfrage** für jede Komponente
   - Analysiere Requirements
   - Entscheidungsbaum
   - Vergleichstabellen
4. **Code-Snippets** (max 20% des Contents)
5. **💡 Deep-Dive Links** zu Core-Konzepten

**Pflicht-Komponenten (Practical-Usage):**
1. **🎯 Was du danach kannst** (ROI-Fokus)
2. **🔄 Before/After Workflows** (Zeit-Vergleich)
3. **🛠️ Sofort-Start** (Step-by-Step)
4. **📊 Erfolgsmessung** (Metriken)
5. **🎯 Action Items** (Heute, diese Woche, langfristig)

---

### Content Quality Standards (Alle Sektionen)

**Pflicht für jedes Kapitel:**
- ✅ Measurable Learning Outcomes
- ✅ Problem-First Approach (was geht ohne schief?)
- ✅ Scaffolded Progression (Intuition → Technical)
- ✅ German Language mit Technical Precision
- ✅ Honest Limitations (was geht NICHT?)
- ✅ Cross-References zu verwandten Themen

**Template-spezifische Tiefe:**
- **A-Sektionen**: Mathematik + Code + Misconceptions + 5-Min-Expert
- **B-Sektionen**: Entscheidungsbäume + Vergleiche + ROI-Kalkulation

---

### Template Organization & Files

Each of the 7 main directories has its own CLAUDE.md file with specific template guidelines:

| Sektion | CLAUDE.md | Template | Tiefe-Level |
|---------|-----------|----------|-------------|
| **01-historical/** | [CLAUDE.md](01-historical/CLAUDE.md) | Template 1 | Deep Technical + Historical Context |
| **02-modern-ai/** | [CLAUDE.md](02-modern-ai/CLAUDE.md) | Template 1 | Deep Technical + SOTA Tracking |
| **03-core/** | [CLAUDE.md](03-core/CLAUDE.md) | Template 1 | **Scaffolded Learning (Vorbild)** |
| **04-advanced/** | [CLAUDE.md](04-advanced/CLAUDE.md) | Template 1 | Deep Technical + Reproduction Focus |
| **05-ethics/** | [CLAUDE.md](05-ethics/CLAUDE.md) | Template 1 | Deep Technical + Social Analysis |
| **06-applications/** | [CLAUDE.md](06-applications/CLAUDE.md) | Template 2 | Decision-Focused (80% Guide, 20% Code) |
| **07-practical-usage/** | [CLAUDE.md](07-practical-usage/CLAUDE.md) | Template 2 | Workflow-Focused (ROI-Orientiert) |

**When working in any directory, refer to that directory's CLAUDE.md file for specific template and writing guidelines.**

---

### Qualitätssicherung: Template-Upgrade 2025-10

**Hintergrund:** Alle theoretischen Templates (A1-A5) wurden auf das Niveau von `03-core/CLAUDE.md` angehoben.

**Vorher vs. Nachher:**

| Aspekt | Vorher | Nachher |
|--------|--------|---------|
| **Mathematik** | Formeln ohne Kontext | Formeln + Intuition + Ableitung + Beispiele |
| **Code** | Teilweise Snippets | Vollständig lauffähig, copy-paste-ready |
| **Misconceptions** | Nicht systematisch | Mindestens 3 pro Kapitel, strukturiert |
| **Testing** | Keine | 5-Minuten-Expert Tests |
| **Benchmarks** | Sporadisch | Systematisch mit Interpretation |
| **Trade-offs** | Implizit | Explizite Analyse |

**Ergebnis:** Alle A-Sektionen (01-05) haben jetzt **gleiche Tiefe** – egal ob historisch, modern oder ethisch.

## Writing Style & Standards

### Writing Style
- **German language** - target audience is German-speaking
- **Educational tone** - you are teaching, not just documenting
- **Historical progression** - show how AI evolved to help humans understand current capabilities
- **Practical relevance** - always connect concepts to real-world applications
- **Honest about limitations** - don't oversell AI capabilities
- **Accessible explanations** - complex concepts explained clearly without losing accuracy
- **Mathematical notation** using LaTeX in Markdown (`$...$` inline, `$$...$$` block)

### Content Principles
- Focus on **understanding over implementation** - help humans grasp concepts, not just copy code
- Provide **historical context** - "why was this developed?" helps humans understand current AI landscape
- **No redundancy** - core concepts centralized in `03-core/` to avoid confusion
- **Progressive complexity** - each chapter builds on previous knowledge
- **Show trade-offs** - help humans understand when to use what and why
- **Connect to ethics** - help humans understand responsible AI use

## Learning Paths

The repository defines several learning paths for different goals:
- **Path 1**: Foundations → Deep Learning (Historical progression)
- **Path 2**: NLP Journey (Text & Language focus)
- **Path 3**: Vision (Computer Vision focus)
- **Path 4**: Practical RAG (Hands-on implementation)
- **Path 5**: Production ML (MLOps focus)
- **Path 6**: Research (State-of-the-art)
- **Path 7**: Ethics & Responsible AI

## Development Notes

This is a documentation-only repository with no build system, tests, or dependencies. Changes involve:
- Adding new chapters following the established template
- Updating existing content for accuracy
- Maintaining cross-references between related concepts
- Ensuring German language consistency
- Following the established notation standards

### Important Structural Notes

- **CRITICAL**: When you add, remove, or rename any folders or files, you MUST update the corresponding `00-overview.md` file in that section
- **Overview files reflect structure**: Each `00-overview.md` should accurately list all subdirectories and files in its section
- **Cross-references**: When moving files between sections, update all cross-references and links
- **Consistency**: Keep file listings in overview files consistent with actual directory structure
- **Examples**:
  - Adding `03-core/training/12-new-topic.md` → Update `03-core/training/00-overview.md`
  - Moving files between sections → Update both source and target overview files
  - Renaming directories → Update all affected overview files and cross-references

When editing, preserve the educational structure and historical progression approach that makes complex ML concepts accessible to German-speaking learners.

## Overview File Standards

Each `00-overview.md` should follow this proven structure from `07-practical-usage/00-overview.md`:

```markdown
# [Section Title]: [Clear Purpose Statement]

## 🎯 Ziel/Lernziele
- Specific, measurable learning outcomes
- Time-based expectations where relevant
- Clear value proposition

## 🧑‍💼 Zielgruppe (for practical sections)
- Primary audience definition
- Prerequisites or background assumptions

## 📖 Philosophie/Geschichte & Kontext
- Why this section exists
- How it connects to the broader AI landscape
- Honest assessment of capabilities and limitations

## 📂 Kapitel in diesem Abschnitt
```
├── 00-overview.md                  (Section overview)
├── 01-topic.md                     (Descriptive name + purpose)
└── XX-topic.md                     (Numbered for logical sequence)
```

## 🎯 Lernpfade (for applicable sections)
- Role-based learning paths with clear progressions

## 🚀 Was du danach kannst
- Concrete, testable skills
- Strategic understanding outcomes

## 🔗 Weiterführende Themen
- Cross-references to related sections
```

## Formatting Standards

**Emoji Navigation System:**
- 🎯 = Goals/Learning Objectives
- 📖 = Context/History/Philosophy
- 🧮 = Theory/Implementation
- 📊 = Comparisons/Benchmarks
- 🔗 = Navigation/Cross-references
- 🚀 = Action Items/Outcomes
- 📂 = File/Chapter Structure
- 🧑‍💼 = Target Audience

**Content Hierarchy:**
- `##` for major sections (using emoji markers above)
- `###` for subsections within major sections
- Consistent indentation in file trees and lists
- Avoid `####` - restructure content instead
