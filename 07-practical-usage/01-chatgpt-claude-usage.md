# ChatGPT & Claude optimal nutzen

## 🎯 Lernziele
- Verstehe die Unterschiede zwischen ChatGPT und Claude
- Lerne effektive Prompting-Strategien für beide Tools
- Verstehe, wann du welches Tool verwenden solltest
- Entwickle professionelle Conversation-Skills mit AI

## 📖 Geschichte & Kontext

**ChatGPT vs. Claude - Die zwei Giganten:**
- **ChatGPT** (OpenAI): Der Pionier, der AI für Millionen zugänglich machte
- **Claude** (Anthropic): Der "sichere" Konkurrent mit Focus auf Helpful, Harmless, Honest

**Warum beide tools?**
Verschiedene Stärken für verschiedene Anwendungsfälle - wie verschiedene Werkzeuge im Werkzeugkasten.

## 🧮 Tool-Vergleich: ChatGPT vs. Claude

### **ChatGPT-Stärken:**
- **Kreativität**: Brainstorming, Creative Writing
- **Code**: Programmierung und Technical Writing
- **Community**: Millionen von Nutzern, viele Prompt-Sammlungen
- **Plugins**: Zugang zu aktuellen Daten, Tools, Web-Browsing
- **Preis**: Günstigere API, kostenlose Version verfügbar

### **Claude-Stärken:**
- **Lange Texte**: Bis zu 200k Token Context Window
- **Sicherheit**: Weniger "Halluzinationen", ehrlichere Antworten
- **Analyse**: Besser für komplexe Dokumenten-Analyse
- **Deutsch**: Oft bessere deutsche Sprachqualität
- **Nuancen**: Versteht subtile Anfragen besser

### **Wann was verwenden?**

| Use Case | ChatGPT | Claude | Warum? |
|----------|---------|--------|--------|
| Brainstorming | ✅ | ❌ | Mehr kreative, wilde Ideen |
| Code schreiben | ✅ | ⚠️ | Größere Community, mehr Beispiele |
| Lange Dokumente analysieren | ❌ | ✅ | 200k vs. 32k Token Context |
| Fakten-Check | ❌ | ✅ | Ehrlicher bei Unwissen |
| Quick Questions | ✅ | ❌ | Schneller, kostengünstiger |
| Business Writing | ⚠️ | ✅ | Professioneller, weniger Risiko |

## 🔬 Prompting Mastery

### **Die 5 Prompting-Prinzipien:**

#### **1. Kontext ist König**
❌ **Schlecht:** "Schreib mir eine E-Mail"
✅ **Besser:** "Schreib eine professionelle E-Mail an einen Kunden, der sich über eine verspätete Lieferung beschwert. Ton: entschuldigend aber lösungsorientiert."

#### **2. Rolle definieren**
✅ **Template:** "Du bist ein [ROLLE] mit [EXPERTISE]. Deine Aufgabe ist [AUFGABE] für [ZIELGRUPPE]."
- "Du bist ein Marketing-Experte mit 10 Jahren E-Commerce Erfahrung..."
- "Du bist ein Lehrer für Grundschüler..."
- "Du bist ein Senior Developer mit Python-Expertise..."

#### **3. Format spezifizieren**
✅ **Beispiele:**
- "Gib mir 5 Bullet Points"
- "Antworte in einer Tabelle mit 3 Spalten"
- "Schreib einen 200-Wort Absatz"
- "Verwende das Format: Problem → Lösung → Nächste Schritte"

#### **4. Beispiele geben (Few-Shot Prompting)**
```
Schreib Social Media Posts im folgenden Stil:

Beispiel 1: "🚀 Neues Feature gelauncht! Jetzt kannst du..."
Beispiel 2: "💡 Pro-Tip: Wusstest du, dass..."

Jetzt schreib einen Post über unser AI-Tool:
```

#### **5. Chain-of-Thought verwenden**
✅ **Bei komplexen Aufgaben:**
"Denk Schritt für Schritt:
1. Analysiere das Problem
2. Identifiziere mögliche Lösungen
3. Bewerte Vor- und Nachteile
4. Gib eine Empfehlung mit Begründung"

### **Advanced Prompting Techniques:**

#### **System Message nutzen (ChatGPT)**
```
System: Du bist ein AI-Assistant für ein deutsches Marketing-Team.
Antworte immer auf Deutsch, sei präzise und business-orientiert.

User: Wie kann ich meine Conversion Rate verbessern?
```

#### **Constitutional AI (Claude)**
```
Ich möchte eine Marketing-Strategie entwickeln.
Bitte sei ehrlich über Limitationen und weise mich auf
ethische Bedenken hin, falls relevant.
```

#### **Iterative Verbesserung**
1. **Erster Prompt:** Grundanfrage
2. **Feedback:** "Mach es präziser / kürzer / detaillierter"
3. **Refinement:** "Fokussiere auf [spezifischen Aspekt]"
4. **Finalisierung:** "Überarbeite für [Zielgruppe]"

## 📊 Conversation Management

### **Multi-Turn Conversations strukturieren:**

#### **1. Session Planning**
- **Topic definieren:** "In dieser Conversation arbeiten wir an..."
- **Ziel setzen:** "Am Ende möchte ich haben..."
- **Grenzen klären:** "Fokussiere auf... , ignoriere..."

#### **2. Context Maintenance**
- **Zusammenfassen:** "Fass unsere bisherige Diskussion zusammen"
- **Referenzieren:** "Basierend auf dem, was wir vorhin über X besprochen haben..."
- **Reset-Punkte:** Neue Conversation für neue Topics

#### **3. Output Management**
- **Versionskontrolle:** "Gib mir 3 verschiedene Varianten"
- **Iteration:** "Version 2: Mach es formeller"
- **Finale Version:** "Gib mir die finale, polierte Version"

### **Conversation Patterns:**

#### **Research Pattern:**
1. "Erkläre mir [Topic] grundlegend"
2. "Was sind die aktuellen Trends in [Topic]?"
3. "Welche Herausforderungen gibt es?"
4. "Gib mir konkrete nächste Schritte"

#### **Problem-Solving Pattern:**
1. "Hier ist mein Problem: [Beschreibung]"
2. "Welche Lösungsansätze gibt es?"
3. "Bewerte diese Optionen: [A, B, C]"
4. "Detailliere die beste Option"

#### **Creation Pattern:**
1. "Ich will [erstellen]. Hier sind meine Anforderungen: [Liste]"
2. "Gib mir einen groben Entwurf"
3. "Verbessere [spezifischen Teil]"
4. "Finalisiere das Ergebnis"

## 📈 Pro-Tips für Alltags-Nutzung

### **Efficiency Hacks:**

#### **Custom Instructions (ChatGPT)**
Setze einmal deine Präferenzen:
```
Über mich: Marketing Manager in einem deutschen B2B SaaS Unternehmen
Antwort-Stil: Präzise, mit konkreten Beispielen, auf Deutsch
Format: Bullet Points bevorzugt, max. 300 Wörter
```

#### **Prompt Libraries aufbauen**
Sammle deine besten Prompts:
- "E-Mail Antworten": [Template]
- "Meeting Zusammenfassung": [Template]
- "Competitor Analysis": [Template]

#### **Keyboard Shortcuts**
- **Copy-Paste Workflow:** Prompt → Copy → Paste in Tool → Work with Output
- **Multiple Tabs:** Verschiedene Conversations für verschiedene Topics
- **Voice Input:** Nutze Spracheingabe für längere Prompts

### **Quality Control:**

#### **Fact-Checking Workflow**
1. **AI-Output erhalten**
2. **Kritische Fakten identifizieren**
3. **Cross-check mit vertrauenswürdigen Quellen**
4. **Bei Unsicherheit: AI fragen "Bist du dir sicher bei..."**

#### **Bias Detection**
- **Perspektivenwechsel:** "Gib mir die Gegenmeinung"
- **Diverse Quellen:** "Berücksichtige auch [andere Perspektive]"
- **Kritische Fragen:** "Welche Annahmen machst du hier?"

## 💼 Business Use Cases

### **1. E-Mail Management**
```
Kontext: Ich bin [Rolle] und muss auf diese E-Mail antworten: [E-Mail]
Ton: [professionell/freundlich/entschuldigend]
Ziel: [Information geben/Meeting vereinbaren/Problem lösen]
Länge: [kurz/mittel/ausführlich]
```

### **2. Meeting Prep**
```
Ich habe ein Meeting über [Topic] mit [Teilnehmer].
Bereite mich vor:
- Agenda-Punkte
- Mögliche Fragen
- Key Arguments
- Next Steps Template
```

### **3. Research & Analysis**
```
Analysiere diesen [Markt/Trend/Competitor].
Gib mir:
- Überblick (200 Wörter)
- Key Insights (5 Punkte)
- Risiken & Chancen
- Handlungsempfehlungen
```

## 🎓 Weiterführende Themen
- **Prompt Engineering Deep-Dive**: [../02-effective-prompting/](../02-effective-prompting/)
- **Workflow Integration**: [../03-workflow-integration/](../03-workflow-integration/)
- **AI Ethics**: [../../05-ethics/](../../05-ethics/) für verantwortlichen AI-Einsatz

## 📚 Ressourcen
- **OpenAI Prompt Examples**: [platform.openai.com/examples](https://platform.openai.com/examples)
- **Claude Prompt Library**: [docs.anthropic.com/claude/prompt-library](https://docs.anthropic.com/claude/prompt-library)
- **Community Prompts**: PromptHero, r/ChatGPT, r/ClaudeAI
- **Advanced Techniques**: Prompt Engineering Guide (GitHub)

---

**🔄 Übung:** Nimm eine Aufgabe aus deinem Arbeitsalltag und entwickle einen optimalen Prompt dafür. Teste ihn mit beiden Tools und vergleiche die Ergebnisse.