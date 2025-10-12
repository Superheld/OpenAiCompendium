# Praktische Projekte: Alles mal ausprobieren!

## 🎯 Ziel
Kleine, praktische Projekte um alle wichtigen ML-Konzepte aus dem Kompendium hands-on zu erleben. Core Concepts werden direkt in den Projekten vermittelt - keine getrennte Theorie.

## 📖 Projekt-Philosophie
- **Core Concepts integriert**: Training, Embeddings, Evaluation, etc. werden praktisch in Projekten gelernt
- **Drei Schwierigkeitsstufen**: 🟢 Easy → 🟡 Medium → 🔴 Hard
- **Core Application Patterns**: Fundamentale ML-Anwendungsmuster werden mit praktischen Projekten verknüpft
- **Hands-on Learning**: Theorie verstehen durch praktisches Ausprobieren
- **Experimentell**: "Wie funktioniert das eigentlich?" praktisch beantworten

## 📂 Thematische Anwendungsbereiche

### **🧠 01-fundamentals/** - Grundbausteine verstehen
Die fundamentalen Konzepte die alle AI-Systeme antreiben

#### **[01-single-neuron.md](01-fundamentals/01-single-neuron.md)** - Das erste künstliche Neuron
**Core Concepts:** Training (Gradient Descent), Aktivierungsfunktionen
- Einzelnes Neuron von Grund auf implementieren
- Logische Operationen (AND, OR, XOR) lernen
- Verstehen warum XOR das Perceptron "gebrochen" hat

#### **[02-embedding-basics.md](01-fundamentals/02-embedding-basics.md)** - Vektorräume verstehen
**Core Concepts:** Embeddings, Similarity Measures
- Word2Vec vs BERT embeddings vergleichen
- Cosine Similarity implementieren
- t-SNE Visualisierung

#### **[03-decision-tree.md](01-fundamentals/03-decision-tree.md)** - Interpretable ML
**Core Concepts:** Model Interpretability, Feature Importance
- Decision Tree von Grund auf bauen
- Feature Importance verstehen
- Random Forest ensemble testen

#### **[04-training-loop.md](01-fundamentals/04-training-loop.md)** - Training Loop von Grund auf
**Core Concepts:** Training (Backpropagation), Optimization
- Forward Pass → Loss → Backward Pass → Update implementieren
- Verstehen wie neuronale Netze wirklich lernen
- Optimizer-Wahl: Adam vs SGD in der Praxis

### **🔍 02-content-understanding/** - Inhalte verstehen & kategorisieren
AI die Texte, Bilder und Inhalte analysiert und einordnet

#### **[02-spam-filter.md](02-content-understanding/02-spam-filter.md)** - Klassisches ML in Aktion
**Core Concepts:** Feature Engineering, Evaluation Metrics
- Naive Bayes Spam-Filter implementieren
- TF-IDF Features verstehen
- Precision, Recall, F1-Score berechnen

#### **[03-cnn-image-classification.md](02-content-understanding/03-cnn-image-classification.md)** - Computer Vision
**Core Concepts:** Computer Vision, Transfer Learning
- CNN für CIFAR-10 implementieren
- Filter-Visualisierung
- Transfer Learning ausprobieren

#### **[04-transformer-attention.md](02-content-understanding/04-transformer-attention.md)** - Language Understanding
**Core Concepts:** Attention, Transformer Architecture
- Self-Attention von Grund auf implementieren
- Attention-Weights visualisieren
- Multi-Head Attention verstehen

#### **[01-classification-systems.md](02-content-understanding/01-classification-systems.md)** - Content Classification ⭐
**Referenz-Implementation:** Automatische Kategorisierung von Inhalten
- Text Classification mit modernen Transformers
- Multi-Label und Hierarchical Classification
- Domain-specific Classification (Legal, Medical, etc.)

### **🎨 03-content-generation/** - Content erstellen
AI die neue Inhalte generiert: Text, Code, Bilder

#### **[01-llm-fine-tuning.md](03-content-generation/01-llm-fine-tuning.md)** - LLM anpassen
**Core Concepts:** Fine-Tuning, PEFT Methods
- LoRA Fine-Tuning ausprobieren
- Before/After Vergleich
- Custom Dataset erstellen

#### **[Text-to-Image Generation]** - KI-Kunst erstellen (geplant)
#### **[Code Generation & Completion]** - AI-gestützte Programmierung (geplant)
#### **[Creative Writing Assistant]** - AI für Content Creation (geplant)

### **🎯 04-personalization/** - Personalisierte Empfehlungen & Suche
AI die individuelle Bedürfnisse versteht und relevante Inhalte findet

#### **[03-basic-rag.md](04-personalization/03-basic-rag.md)** - Knowledge-basierte AI
**Core Concepts:** Embeddings, Vector Search, Generation, RAG Pipeline
- Embedding-basierte Dokumentensuche
- Chunking-Strategien vergleichen
- LLM für Antwort-Generation

#### **[01-rag-systems.md](04-personalization/01-rag-systems.md)** - Retrieval-Augmented Generation ⭐
**Referenz-Implementation:** Knowledge-basierte AI-Systeme
- Dense vs. Sparse Embedding-Strategien
- Advanced RAG Patterns und Optimization
- Production-ready RAG Architecture

#### **[02-search-systems.md](04-personalization/02-search-systems.md)** - Information Retrieval ⭐
**Referenz-Implementation:** Semantische und hybride Suchsysteme
- Classical vs. Neural Search
- Hybrid Dense+Sparse Retrieval
- Query Understanding und Result Ranking

#### **[Recommendation Systems]** - Personalisierte Empfehlungen (geplant)
#### **[Adaptive User Interfaces]** - UI die sich anpasst (geplant)

### **🤖 05-automation/** - Autonome Systeme & Agents
AI die selbstständig handelt und komplexe Aufgaben übernimmt

#### **[01-agentic-ai-system.md](05-automation/01-agentic-ai-system.md)** - Autonome AI-Agenten
**Core Concepts:** Agentic AI, Planning, Tool Use
- Multi-step reasoning implementieren
- Tool-calling und API-Integration
- Agent-Memory und Kontext-Management

#### **[02-multimodal-system.md](05-automation/02-multimodal-system.md)** - Vision + Language
**Core Concepts:** Multimodal AI, Cross-Modal Learning
- Image + Text Embeddings kombinieren
- CLIP-style Model implementieren
- Visual Question Answering

#### **[Workflow Automation]** - AI für Business Prozesse (geplant)
#### **[API-calling Agents]** - AI die Services nutzt (geplant)

### **⚖️ 06-safety-ethics/** - Verantwortliche AI
AI-Sicherheit, Fairness und ethische Aspekte

#### **[01-bias-detection.md](06-safety-ethics/01-bias-detection.md)** - AI Ethics praktisch
**Core Concepts:** Ethics, Fairness Metrics
- Gender/Racial Bias in Word Embeddings finden
- Fairness-Metriken implementieren
- Debiasing-Techniken testen

#### **[02-hallucination-detection.md](06-safety-ethics/02-hallucination-detection.md)** - LLM-Probleme verstehen
**Core Concepts:** Evaluation, Quality Assessment
- Hallucinations provozieren und erkennen
- Consistency-Checking implementieren
- Fact-Verification testen

#### **[Adversarial Robustness]** - AI gegen Angriffe (geplant)
#### **[Privacy-Preserving ML]** - Datenschutz-konforme AI (geplant)

### **🏗️ 07-infrastructure/** - Production & Scaling
AI-Systeme in Production betreiben und skalieren

#### **[02-vector-database.md](07-infrastructure/02-vector-database.md)** - Similarity Search at Scale
**Core Concepts:** Infrastructure, Vector Indexing
- ChromaDB vs Qdrant vergleichen
- HNSW Index-Performance testen
- Million+ Embeddings indexieren

#### **[03-distributed-training.md](07-infrastructure/03-distributed-training.md)** - Scaling ML
**Core Concepts:** Distributed Training, Optimization
- Multi-GPU Training setup
- Data vs Model Parallelism
- Performance-Benchmarks

#### **[04-production-monitoring.md](07-infrastructure/04-production-monitoring.md)** - ML in Production
**Core Concepts:** Infrastructure, Monitoring, MLOps
- Model Drift Detection
- Performance Monitoring Setup
- A/B Testing implementieren

#### **[01-model-selection.md](07-infrastructure/01-model-selection.md)** - Model Selection & Evaluation ⭐
**Referenz-Implementation:** Das richtige Model für den Use Case finden
- Benchmark-driven Model Selection
- Performance vs. Cost Trade-offs
- A/B Testing von ML Models

## 🛠️ Technische Setup-Empfehlungen

### **🟢 Easy Projekte (01-05):**
- Python + Jupyter Notebooks
- NumPy, Pandas, Matplotlib
- Scikit-learn für Vergleiche

### **🟡 Medium Projekte (06-11):**
- PyTorch oder TensorFlow
- Hugging Face Transformers
- Weights & Biases für Logging

### **🔴 Hard Projekte (12-16):**
- Docker für Containerization
- Git LFS für große Modelle
- Cloud Credits (AWS/GCP) für Scaling

## 🎓 Lern-Ansatz

**Für jedes Projekt:**

1. **📚 Theorie zuerst**: Lies das entsprechende Kompendium-Kapitel
2. **🔬 Minimal-Implementation**: Baue die einfachste Version
3. **🔍 Experimentieren**: "Was passiert wenn...?" - verschiedene Parameter testen
4. **📝 Dokumentieren**: Was hast du gelernt? Was war überraschend?
5. **🔗 Verbindungen**: Wie hängt es mit anderen Projekten zusammen?

**Ziel:** Nicht perfekte Systeme bauen, sondern **verstehen wie alles funktioniert** und dabei alle Core Concepts praktisch lernen!

## 🎯 Lernpfade

### **🎯 Systematischer Aufbau:**
🟢 Easy: 01 → 03 → 02 → 04 → 05
🟡 Medium: 06 → 08 → 09 → 10 → 11
🔴 Hard: 12 → 14 → 15 → 16 → 13

### **🎲 Nach Interesse:**
- **Historical ML Fan:** 01 → 02 → 04 → 06 → 08
- **Modern AI Focus:** 05 → 09 → 12 → 16
- **Ethics & Safety:** 10 → 11 → 15
- **Infrastructure Nerd:** 13 → 14 → 15
- **Experimentierfreudig:** Einfach das nehmen was gerade interessant aussieht! 😊

## 🚀 Was du danach kannst
- Du hast alle wichtigen ML-Konzepte hands-on implementiert
- Du verstehst Core Application Patterns (RAG, Search, Classification)
- Du kannst AI-Projekte von der Idee bis zur Production entwickeln
- Du hast praktische Erfahrung mit verschiedenen Schwierigkeitsstufen

### **🎖️ Core Concepts Praktisch Beherrscht:**
- **Training:** 01, 06, 09, 13
- **Embeddings:** 03, 05, 14, 16
- **Evaluation:** 02, 10, 11, 15
- **Ethics:** 10, 11, 12
- **Infrastructure:** 13, 14, 15

## 🔗 Weiterführende Themen
- **Theoretische Fundamente**: [../03-core/](../03-core/) für tieferes Verständnis
- **Research & Innovation**: [../04-advanced/](../04-advanced/) für cutting-edge Techniken
- **AI im Alltag**: [../07-practical-usage/](../07-practical-usage/) für Tool-Nutzung
- **Ethics in Practice**: [../05-ethics/](../05-ethics/) für verantwortliche AI-Entwicklung