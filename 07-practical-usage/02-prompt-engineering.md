# Prompt Engineering: Die Grundlagen

## 🎯 Lernziele
- **Prompt-Anatomie** verstehen und anwenden
- **Zero-Shot, Few-Shot und Chain-of-Thought** Techniken beherrschen
- **Kontext, Rolle und Constraints** strategisch einsetzen
- **Prompt-Iteration** systematisch durchführen
- **Business-relevante Prompt-Patterns** entwickeln

## 📖 Geschichte & Kontext

**Von Command-Line zu Natural Language:**
Prompt Engineering ist die moderne Evolution der Computer-Bedienung. Während wir früher präzise Befehle tippen mussten (`cd /Users/folder`), können wir heute AI in natürlicher Sprache instruieren - aber die Präzision ist immer noch entscheidend.

**Der Paradigmenwechsel:**
- **Traditional Programming**: Wir schreiben Code, der exakt ausgeführt wird
- **Prompt Engineering**: Wir instruieren AI, die interpretiert und kreiert
- **Neue Skill**: Nicht mehr "Wie programmiere ich das?" sondern "Wie erkläre ich das der AI?"

**Warum Prompt Engineering Business-kritisch ist:**
- **Zeit-Effizienz**: 10 Minuten besseres Prompting sparen Stunden Nacharbeit
- **Qualitäts-Konsistenz**: Gute Prompts = reproduzierbare Ergebnisse
- **Skalierbarkeit**: Einmal entwickelte Prompts können teamweit genutzt werden

## 🧮 Theorie: Systematisches Prompt-Design

### Die 6 Säulen eines starken Prompts

#### 1. **Kontext (Context Setting)**
```markdown
**Zweck**: AI das "Warum" und "Wo" vermitteln
**Beispiel**:
"Du hilfst einem B2B-SaaS-Startup dabei, ihre erste Marketing-Kampagne zu planen.
Das Unternehmen hat 50 Mitarbeiter, verkauft Projektmanagement-Software an
Agenturen und hat ein monatliches Marketing-Budget von €15.000."
```

#### 2. **Rolle (Role Assignment)**
```markdown
**Zweck**: AI-Expertise und Perspektive definieren
**Beispiel**:
"Du bist ein Senior Marketing Manager mit 10 Jahren Erfahrung im B2B-Bereich
und spezieller Expertise in SaaS-Marketing und Growth Hacking."
```

#### 3. **Aufgabe (Task Definition)**
```markdown
**Zweck**: Präzise definieren, was gemacht werden soll
**Beispiel**:
"Entwickle eine 90-Tage-Marketing-Strategie mit konkreten Kanälen,
Budget-Allocation und messbaren KPIs."
```

#### 4. **Format (Output Structure)**
```markdown
**Zweck**: Gewünschte Ausgabe-Struktur vorgeben
**Beispiel**:
"Strukturiere deine Antwort als:
1. Executive Summary (3 Sätze)
2. Kanal-Mix (Tabelle mit Budget%)
3. 90-Tage-Timeline (Wochenweise)
4. KPIs & Tracking (5 Metriken)"
```

#### 5. **Beispiele (Few-Shot Learning)**
```markdown
**Zweck**: Stil und Qualität durch Beispiele demonstrieren
**Beispiel**:
"Hier ist ein Beispiel für den gewünschten Stil:
'Woche 1-2: Content-Audit + Persona-Research (Budget: €1.000)
- LinkedIn-Content-Kalender erstellen
- 3 Buyer-Interviews führen
- Competitor-Analysis dokumentieren'"
```

#### 6. **Constraints (Guardrails)**
```markdown
**Zweck**: Unwünschte Outputs vermeiden
**Beispiel**:
"Constraints:
- Keine unrealistischen Wachstumszahlen (>20% MoM)
- Fokus auf messbare, nicht auf 'Brand Awareness' Metriken
- Berücksichtige kleine Team-Kapazität
- Vermeide teure Tools (>€500/Monat)"
```

### Prompt-Engineering-Techniken

#### **Zero-Shot Prompting**
```markdown
**Definition**: AI ohne Beispiele instruieren
**Wann nutzen**: Einfache, standardisierte Aufgaben
**Business-Beispiel**:
"Schreibe eine professionelle Follow-up-E-Mail nach einem Sales-Call.
Der Kunde war interessiert, möchte aber erst nächstes Quartal entscheiden."

**Vorteile**: Schnell, direkt
**Nachteile**: Weniger Kontrolle über Output-Qualität
```

#### **Few-Shot Prompting**
```markdown
**Definition**: AI durch 2-5 Beispiele das gewünschte Muster zeigen
**Wann nutzen**: Konsistente Formate, spezielle Stil-Anforderungen

**Business-Beispiel**:
"Schreibe LinkedIn-Posts im folgenden Stil:

Beispiel 1:
'🚀 Gerade unser Q3-Review abgeschlossen.
3 Learnings für andere SaaS-Founder:
• Customer Success > Customer Acquisition
• Team-Happiness korreliert direkt mit Retention
• Weniger Features, bessere UX
Was sind eure Q3-Learnings? 👇'

Beispiel 2:
'📊 Spannende Studie gelesen: 67% der B2B-Buyer recherchieren >6 Monate.
Bedeutet für uns:
• Content muss Long-Tail-Keywords abdecken
• Nurturing-Sequences sind business-critical
• Sales-Cycle-Verständnis = Competitive Advantage
Wie lange sind eure Sales-Cycles? 🤔'

Jetzt schreibe einen Post über [DEIN THEMA] in diesem Stil."
```

#### **Chain-of-Thought Prompting**
```markdown
**Definition**: AI Schritt-für-Schritt durch komplexe Probleme führen
**Wann nutzen**: Komplexe Analysen, strategische Entscheidungen

**Business-Beispiel**:
"Analysiere, ob wir in den französischen Markt expandieren sollten.
Gehe dabei systematisch vor:

Schritt 1: Markt-Analyse
- Marktgröße für Projektmanagement-Software in Frankreich
- Hauptwettbewerber und deren Marktanteile
- Regulatorische Besonderheiten (DSGVO, arbeitsrechtliche Aspekte)

Schritt 2: Ressourcen-Assessment
- Welche internen Kapazitäten brauchen wir?
- Lokalisierungs-Aufwand (Sprache, Kultur, Support)
- Geschätzte Investition für Jahr 1

Schritt 3: Risiko-Bewertung
- Was könnte schiefgehen?
- Welche Annahmen sind kritisch?
- Exit-Strategie falls es nicht funktioniert

Schritt 4: Go/No-Go Empfehlung
- Klare Empfehlung mit Begründung
- Nächste Schritte für beide Szenarien
- Timeline und Milestones

Denke laut durch jeden Schritt und zeige deine Überlegungen."
```

## 🔬 Implementierung: Business-Prompt-Patterns

### Pattern 1: Der "Executive Briefing" Prompt

```markdown
**Use Case**: Komplexe Informationen für C-Level aufbereiten

**Template**:
```
Du bist ein erfahrener Management Consultant.

Kontext: [SITUATION/PROBLEM BESCHREIBUNG]

Erstelle ein Executive Briefing für [ZIELGRUPPE] über [THEMA].

Format:
• **Situation**: Was ist los? (2 Sätze)
• **Impact**: Warum wichtig? (€/Zeit/Risiko quantifiziert)
• **Options**: 3 Handlungsoptionen mit Pros/Cons
• **Recommendation**: Klare Empfehlung + Begründung
• **Next Steps**: 3 konkrete Maßnahmen mit Timeframes

Stil: Fakten-basiert, executive-ready, actionable
Länge: Maximal 200 Wörter
```

**Praxis-Beispiel**:
```
Du bist ein erfahrener Management Consultant.

Kontext: Unser E-Commerce-Shop hat 25% höhere Retour-Rate als der Branchenschnitt.
Das kostet uns monatlich ca. €50.000 in Logistik und entgangenen Umsätzen.

Erstelle ein Executive Briefing für unseren CEO über dieses Problem.

[Rest wie Template...]
```
```

### Pattern 2: Der "Content Factory" Prompt

```markdown
**Use Case**: Konsistenten Content in großen Mengen produzieren

**Template**:
```
Du bist ein Content Marketing Specialist für [BRANCHE].

Aufgabe: Erstelle [ANZAHL] [CONTENT-TYP] zu [THEMA].

Content-Specs:
• Zielgruppe: [PERSONA BESCHREIBUNG]
• Ton: [BRAND VOICE]
• Länge: [WORD COUNT]
• CTA: [CALL TO ACTION]
• Keywords: [SEO BEGRIFFE]

Format-Vorgabe:
1. Hook (erste 2 Sätze)
2. Problem/Pain Point
3. Solution/Value Prop
4. Social Proof/Example
5. CTA

Beispiel für gewünschten Stil:
[MUSTER-CONTENT EINFÜGEN]

Variiere Hooks und Angles, aber behalte Struktur bei.
```

**Praxis-Beispiel**:
```
Du bist ein Content Marketing Specialist für B2B-SaaS.

Aufgabe: Erstelle 5 LinkedIn-Posts zu "Remote Team Management".

Content-Specs:
• Zielgruppe: Team-Leads in Tech-Unternehmen (25-45 Jahre)
• Ton: Authentisch, pragmatisch, leicht humorvoll
• Länge: 100-150 Wörter
• CTA: Kommentar mit eigener Erfahrung
• Keywords: Remote Work, Team Management, Produktivität

[Rest wie Template...]
```
```

### Pattern 3: Der "Analysis Engine" Prompt

```markdown
**Use Case**: Daten/Dokumente systematisch analysieren

**Template**:
```
Du bist ein Senior [DOMAIN] Analyst mit [EXPERTISE].

Analyse-Auftrag: [DOKUMENT/DATEN] analysieren bezüglich [FRAGESTELLUNG].

Analyse-Framework:
1. **Datenqualität**: Vollständigkeit, Verlässlichkeit, Aktualität
2. **Patterns**: Trends, Auffälligkeiten, Korrelationen
3. **Insights**: 3-5 wichtigste Erkenntnisse
4. **Implications**: Was bedeutet das für [BUSINESS CONTEXT]?
5. **Action Items**: Konkrete nächste Schritte

Anforderungen:
• Quantifiziere wo möglich (%, €, Zeiträume)
• Unterscheide zwischen Facts und Interpretationen
• Priorisiere nach Business-Impact
• Nenne Limitationen deiner Analyse

Output-Format: Executive Summary + Detailanalyse
```

**Praxis-Beispiel**:
```
Du bist ein Senior Marketing Analyst mit 8 Jahren B2B-SaaS-Erfahrung.

Analyse-Auftrag: Unsere Q3-Marketing-Performance analysieren bezüglich ROI und Kanal-Effektivität.

Daten:
• Google Ads: €15.000 investiert, 450 Leads, 12 Customers
• LinkedIn Ads: €8.000 investiert, 180 Leads, 8 Customers
• Content Marketing: €5.000 investiert, 320 Leads, 15 Customers
• Events: €12.000 investiert, 90 Leads, 6 Customers

[Rest wie Template...]
```
```

### Pattern 4: Der "Decision Framework" Prompt

```markdown
**Use Case**: Komplexe Business-Entscheidungen strukturiert angehen

**Template**:
```
Du bist ein erfahrener [ROLLE] und hilfst bei einer wichtigen Entscheidung.

Decision Context: [SITUATION & ENTSCHEIDUNGSGEGENSTAND]

Verwende dieses Framework:

**STEP 1: Problem Definition**
• Was genau entscheiden wir?
• Welche Stakeholder sind betroffen?
• Was passiert, wenn wir NICHT entscheiden?

**STEP 2: Criteria Definition**
• Welche Faktoren sind entscheidungsrelevant?
• Wie gewichten wir diese? (Must-have vs. Nice-to-have)

**STEP 3: Option Generation**
• Mindestens 3 realistische Optionen
• Include "Status Quo" und "Do Nothing"

**STEP 4: Option Evaluation**
• Jede Option gegen alle Kriterien bewerten
• Pros/Cons mit konkreten Beispielen
• Risiken und Mitigation-Strategien

**STEP 5: Recommendation**
• Klare Empfehlung mit Confidence Level (%)
• Top 3 Gründe für diese Wahl
• Implementation Roadmap

Denke strukturiert und zeige deine Überlegungen.
```
```

## 📊 Vergleiche & Varianten

### Prompt-Länge: Wann welcher Ansatz?

| Prompt-Typ | Länge | Wann nutzen | Business-Beispiel |
|------------|-------|-------------|-------------------|
| **Micro-Prompt** | <20 Wörter | Quick Tasks, Known Formats | "Formuliere diese E-Mail professioneller: [TEXT]" |
| **Standard-Prompt** | 50-150 Wörter | Meiste Business-Tasks | Content-Erstellung, Standard-Analysen |
| **Macro-Prompt** | 200+ Wörter | Komplexe Tasks, Neue Formate | Strategische Analysen, Custom Workflows |
| **Mega-Prompt** | 500+ Wörter | Vollständige Systeme | AI-Assistenten-Konfiguration, Komplette Workflows |

### Prompt-Evolution: Von Basic zu Advanced

#### **Level 1: Basic (Anfänger)**
```
"Schreib einen Blogpost über AI im Marketing"
```

#### **Level 2: Structured (Fortgeschritten)**
```
Du bist ein Marketing-Experte. Schreibe einen 800-Wort-Blogpost über
"AI-Tools für Marketing-Automatisierung". Zielgruppe: CMOs in
mittelständischen Unternehmen. Struktur: Problem → Solution → Use Cases → ROI.
```

#### **Level 3: Systemic (Profi)**
```
**Rolle**: Senior Marketing Strategist mit 10 Jahren AI/MarTech-Erfahrung

**Context**: Deutsche Mittelstands-CMOs (50-500 MA) evaluieren AI-Tools,
sind aber skeptisch wegen Komplexität und unsicherem ROI.

**Task**: Schreibe einen vertrauensbildenden Blogpost über praktische
AI-Marketing-Tools mit deutschen Fallstudien.

**Format**:
• Headlines: Action-oriented, keine Superlative
• Struktur: Hook → Status Quo Pain → Tool-Categories → German Case Studies → ROI-Calculator → Next Steps
• Länge: 1.200 Wörter
• Tone: Pragmatisch, nicht zu "techy", vertrauenserweckend

**Constraints**:
• Keine US-only Tools
• DSGVO-Compliance erwähnen
• Realistische ROI-Zahlen (keine "500% Steigerung")
• Mindestens 2 deutsche Unternehmen als Beispiele

**Output-Check**:
Würde sich ein skeptischer Mittelstands-CMO nach dem Lesen
trauen, ein AI-Tool zu testen?
```

### Prompt-Testing: Systematische Verbesserung

#### **A/B-Testing für Prompts**
```markdown
**Version A** (Direkt):
"Analysiere unsere Kundenfeedbacks und finde Verbesserungsmöglichkeiten."

**Version B** (Rolle + Struktur):
"Du bist ein Customer Experience Analyst. Analysiere die 47 Feedbacks
systematisch: 1) Kategorisiere Themes, 2) Priorisiere nach Impact,
3) Schlage konkrete Maßnahmen vor. Format: Executive Summary + Detailanalyse."

**Bewertung**: Version B produziert strukturiertere, actionable Ergebnisse
```

#### **Iterative Verbesserung**
```markdown
**Round 1**: Basic Prompt → Ergebnis analysieren
**Round 2**: + Rolle und Kontext → Bessere Relevanz
**Round 3**: + Beispiele und Format → Konsistente Struktur
**Round 4**: + Constraints und Guardrails → Verfeinerte Outputs
**Round 5**: + Output-Validation → Production-ready Prompt
```

## 🎓 Weiterführende Themen

### Advanced Prompting Techniques
- **Constitutional AI**: Ethische Guidelines in Prompts einbauen
- **Recursive Prompting**: AI zur Selbst-Verbesserung nutzen
- **Meta-Prompting**: Prompts, die bessere Prompts generieren
- **Multi-Agent Prompting**: Verschiedene AI-Rollen interagieren lassen

### Tool-Integration
- **Custom Instructions**: Persönliche AI-Assistenten konfigurieren
- **API-Prompting**: Prompts für automatisierte Workflows
- **Prompt Chaining**: Komplexe Tasks in Sequenzen aufteilen

### Measuring Prompt Performance
- **Consistency Metrics**: Wie reproduzierbar sind Ergebnisse?
- **Quality Scoring**: Objektive Bewertungskriterien entwickeln
- **Time-to-Value**: Effizienz-Metriken für Prompt-Optimierung

## 📚 Ressourcen

### Prompt-Libraries für Business
- **Sales Prompts**: Prospecting, Follow-ups, Objection Handling
- **Marketing Prompts**: Content, Campaigns, Analysis
- **Operations Prompts**: Process Documentation, Training Materials
- **Strategy Prompts**: Planning, Analysis, Decision Support

### Tools für Prompt Development
- **PromptPerfect**: Automatische Prompt-Optimierung
- **LangSmith**: Prompt-Versioning und Testing
- **Promptfoo**: Open-Source Prompt-Testing Framework

### Weiterbildung
- **OpenAI Cookbook**: Technische Prompt-Beispiele
- **Anthropic Claude Guide**: Best Practices für Claude
- **Learn Prompting**: Strukturierter Online-Kurs

---

**💡 Next Steps**:
1. Teste den "Executive Briefing" Prompt mit einem aktuellen Problem
2. Entwickle 3 eigene Prompt-Templates für deine häufigsten Tasks
3. Implementiere systematisches A/B-Testing für deine wichtigsten Prompts

**🔗 Weiter zu**: [02-conversation-design.md](02-conversation-design.md) für Multi-Turn-Dialog-Strategien