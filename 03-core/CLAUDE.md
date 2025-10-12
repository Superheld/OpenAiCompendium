# Template für 03-core/

## Template 1: Core Concepts (Didaktisch optimiert)
**Für:** Zeitlose Fundamentals und Kernkonzepte

**Philosophie:** Tiefes, systematisches Verständnis fundamentaler ML-Konzepte durch **scaffolded learning** - von Intuition zu Formalismus, von Problem zu Lösung.

**Didaktische Prinzipien:**
- **Problem-First**: Zeige WARUM wir das Konzept brauchen (was geht ohne schief?)
- **Scaffolded Progression**: Intuition → Visualisierung → Mathematik → Implementation
- **Misconception Debugging**: Adressiere häufige Missverständnisse explizit
- **Active Learning**: Hands-on Experimente und 5-Minute Expert Tests
- **Understanding over Memorization**: Erkläre das WARUM, nicht nur das WAS

```markdown
# [Core Concept]: [Eingängiger problem-orientierter Titel]

## ❓ Das Problem (Problem-First)
**Ohne [Konzept] geht folgendes schief:**
- [Konkrete Failure Scenario 1 - real-world Beispiel]
- [Konkrete Failure Scenario 2 - technisches Problem]
- [Konkrete Failure Scenario 3 - praktische Konsequenz]

**Die zentrale Frage:**
[Eine Frage die Lernende wirklich haben - macht das Kapitel relevant]

**Beispiel-Szenario:**
[Konkretes Szenario das zeigt warum wir dieses Konzept brauchen]

## 🎯 Lernziele
Nach diesem Kapitel kannst du:
- [ ] [Testbares Verständnis-Ziel - "Du verstehst..."]
- [ ] [Testbares Anwendungs-Ziel - "Du kannst implementieren..."]
- [ ] [Testbares Entscheidungs-Ziel - "Du wählst richtig zwischen..."]

## 🧠 Intuition zuerst (Scaffolded Progression)

### Alltagsanalogie
[Nicht-technische Analogie die jeder versteht]

**Beispiel aus dem echten Leben:**
[Konkretes Beispiel mit dem sich Menschen identifizieren können]

### Visualisierung: Von 2D zu hochdimensional
[Einfache 2D/3D Darstellung zum Verstehen]

**Was bedeutet das für echte Systeme?**
[Brücke von Visualisierung zur Realität]

### Die Brücke zur Mathematik
[Übergang: "Jetzt machen wir das präzise - aber die Intuition bleibt gleich"]

## 🧮 Das Konzept verstehen

### Mathematische Grundlagen
[Formeln MIT Erklärung WARUM, nicht nur WAS]

**Intuition hinter der Formel:**
[Was bedeutet diese Formel wirklich?]

**Schritt-für-Schritt Ableitung:**
[Wie kommt man zu dieser Formel?]

### Algorithmus: So funktioniert es
[Implementation aufgedröselt mit Kommentaren]

```python
# Code mit inline-Erklärungen WARUM jeder Schritt nötig ist
```

**Warum dieser Ansatz?**
[Design-Entscheidungen erklären]

### Varianten & Trade-offs
| Variante | Vorteil | Nachteil | Wann nutzen? |
|----------|---------|----------|--------------|
| [Var 1]  | [...]   | [...]    | [...]        |
| [Var 2]  | [...]   | [...]    | [...]        |

## ⚠️ Häufige Missverständnisse (Misconception Debugging)

### ❌ Missverständnis 1: "[Was Leute oft falsch denken]"
**Warum das falsch ist:** [Erklärung mit Beispiel]
**✓ Richtig ist:** [Korrektur]
**Merksatz:** [Einprägsame Formulierung]

### ❌ Missverständnis 2: "[Weitere common misconception]"
**Warum das falsch ist:** [Erklärung]
**✓ Richtig ist:** [Korrektur]
**Merksatz:** [Einprägsame Formulierung]

### ❌ Missverständnis 3: "[Edge case den viele übersehen]"
**Warum das falsch ist:** [Erklärung]
**✓ Richtig ist:** [Korrektur]
**Merksatz:** [Einprägsame Formulierung]

## 🔬 Hands-On: [Praktisches Mini-Experiment]
[Kurzer Code zum Selbst-Ausprobieren - copy-paste-ready]

```python
# Vollständiges lauffähiges Beispiel
# Mit Kommentaren was zu beobachten ist
```

**Was du beobachten solltest:**
- [Erwartetes Ergebnis 1]
- [Erwartetes Ergebnis 2]

**Experimentiere selbst:**
- Was passiert wenn du [Parameter X] änderst?
- Wie verhält es sich bei [Edge Case Y]?

**Erwartung vs. Realität:**
[Was könnte überraschen und warum?]

## ⏱️ 5-Minuten-Experte
Teste dein Verständnis - kannst du diese ohne nachzuschauen beantworten?

### 1. Verständnisfrage: [Grundlegendes Konzept]
<details><summary>💡 Zeige Antwort</summary>

**Antwort:** [Kurze prägnante Antwort]

**Erklärung:** [Warum ist das so?]

**Merksatz:** [Einprägsame Formulierung]
</details>

### 2. Anwendungsfrage: [Praktisches Szenario]
<details><summary>💡 Zeige Antwort</summary>

**Antwort:** [Was würdest du tun?]

**Begründung:** [Warum ist das die richtige Entscheidung?]

**Alternative:** [Was wäre auch okay und wann?]
</details>

### 3. Trade-off-Frage: [Entscheidungssituation]
<details><summary>💡 Zeige Antwort</summary>

**Antwort:** [Differenzierte Antwort]

**Kontext matters:** [Wann gilt was?]

**Red Flags:** [Wann ist die andere Option besser?]
</details>

## 📊 Vergleiche & Varianten

### Wann nutze ich was?

| Use Case | Empfehlung | Warum? | Trade-off |
|----------|------------|--------|-----------|
| [Szenario 1] | [Option A] | [Begründung] | [Was opferst du?] |
| [Szenario 2] | [Option B] | [Begründung] | [Was opferst du?] |
| [Szenario 3] | [Option C] | [Begründung] | [Was opferst du?] |

### Decision Tree

```
Brauchst du [Feature X]?
├─ Ja → [Lösung A]
│   └─ Performance wichtiger als Genauigkeit? → [Variante A1]
└─ Nein → [Lösung B]
    └─ Große Datenmengen? → [Variante B1]
```

## 🛠️ Tools & Frameworks

### Wichtigste Libraries
[Konkrete Tools mit Begründung WARUM diese]

**Code-Beispiel:**
```python
# Production-ready Code-Snippet
# Mit Best Practices und Error Handling
```

**Häufige Stolpersteine:**
- [Problem 1 und Lösung]
- [Problem 2 und Lösung]

## 🚀 Was du jetzt kannst

**Verständnis:**
- ✓ Du verstehst [Kernkonzept] von Intuition bis Mathematik
- ✓ Du erkennst [häufigen Fehler] und weißt wie du ihn vermeidest
- ✓ Du siehst [Anwendungsmöglichkeiten] in verschiedenen Bereichen

**Praktische Fähigkeiten:**
- ✓ Du implementierst [Konzept] from scratch
- ✓ Du wählst zwischen [Varianten] basierend auf Requirements
- ✓ Du debuggst [typische Probleme] in Production

**Nächste Schritte:**
- [ ] [Praktisches Experiment zum Vertiefen]
- [ ] [Weiterführendes Kapitel]
- [ ] [Real-world Anwendung]

## 🔗 Weiterführende Themen

**Nächster logischer Schritt:**
→ [Kapitel das direkt aufbaut]

**Vertiefung:**
→ [Advanced Topics für Interessierte]

**Praktische Anwendung:**
→ [Wo wird das in Production genutzt?]

**Verwandte Konzepte:**
- [Konzept A] - [Wie hängt es zusammen?]
- [Konzept B] - [Unterschied/Gemeinsamkeit]
```

**Besonderheiten für Core:**
- **Systematisch vollständig**: Alle Aspekte eines Konzepts
- **Didaktisch optimiert**: Problem → Intuition → Formalisierung → Anwendung
- **Misconception Debugging**: Häufige Fehler explizit adressieren
- **Active Learning**: Hands-on + 5-Minute Expert Tests
- **Tool-Agnostic**: Prinzipien vor spezifischen Implementierungen
- **Cross-Domain**: Anwendung über verschiedene ML-Bereiche
- **Single Source of Truth**: Definitive Behandlung des Themas