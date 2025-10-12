# Hallucination Detection & Prevention

## 🎯 Lernziele
- Verstehe was Hallucinations in LLMs sind und warum sie entstehen
- Lerne verschiedene Detection-Methoden für Hallucinations
- Verstehe Prevention-Strategien und Mitigation-Techniken
- Lerne praktische Tools für Hallucination-Monitoring

## 📖 Geschichte & Kontext

**Hallucinations** sind eines der größten Probleme moderner LLMs - sie generieren plausibel klingende, aber faktisch falsche Informationen.

**Problem-Evolution:**
- **2019-2020**: Erste Beobachtungen bei GPT-2/GPT-3
- **2021-2022**: Systematische Forschung zu Hallucination-Typen
- **2023-2024**: Detection & Prevention-Tools für Production
- **2024+**: Integration in RAG-Systeme und Enterprise AI

**Warum kritisch?**
- ❌ **Factual Errors**: Falsche Fakten, Daten, Zahlen
- ❌ **Invented Sources**: Nicht-existente Papers, Links, Zitate
- ❌ **Inconsistency**: Widersprüchliche Aussagen
- ❌ **User Trust**: Untergräbt Vertrauen in AI-Systeme

## 🧮 Theorie

### Was sind Hallucinations?

**Definition:** LLM generiert Output, der nicht durch Input oder Training-Daten gestützt ist.

**Typen von Hallucinations:**

#### 1. **Factual Hallucinations**
```
Query: "When was Einstein born?"
Correct: "Albert Einstein was born on March 14, 1879"
Hallucination: "Albert Einstein was born on April 18, 1875"
```

#### 2. **Source Hallucinations**
```
Query: "Give me a paper about X"
Hallucination: "According to Smith et al. (2023) in 'Advanced Methods for X'..."
→ Paper existiert nicht!
```

#### 3. **Contextual Hallucinations (RAG-specific)**
```
Context: "Product A costs $100"
Query: "How much does Product B cost?"
Hallucination: "Based on the information provided, Product B costs $150"
→ Nicht im Context enthalten!
```

#### 4. **Temporal Hallucinations**
```
Query: "Latest developments in AI 2024"
Hallucination: Erfindet Events, die nach Training-Cutoff liegen
```

### Ursachen von Hallucinations

#### 1. **Training Data Issues**
```
- Inkonsistente/falsche Daten im Training-Set
- Bias in Datenquellen
- Temporal Mismatch (veraltete Informationen)
```

#### 2. **Model Architecture**
```
- Autoregressive Generation (jedes Token basiert auf vorherigen)
- Attention Patterns können "falsch fokussieren"
- Overfitting auf Training-Patterns
```

#### 3. **Generation Parameters**
```python
# Höhere Temperature → mehr Hallucinations
response = llm.generate(
    prompt,
    temperature=0.9,  # ❌ Höhere Kreativität, mehr Hallucinations
    top_p=0.95        # ❌ Mehr diverse Tokens
)

# Niedrigere Temperature → weniger Hallucinations
response = llm.generate(
    prompt,
    temperature=0.1,  # ✅ Deterministischer
    top_p=0.8         # ✅ Fokussierter auf wahrscheinliche Tokens
)
```

#### 4. **Context Issues (RAG)**
```
- Irrelevanter/unvollständiger retrieved Context
- Misleading Chunks
- Context-Query Mismatch
```

## 🔬 Detection-Methoden

### 1. **Consistency Checking**

#### **Self-Consistency**
```python
def detect_hallucination_consistency(query: str, model, n_samples: int = 5) -> dict:
    """
    Generate multiple responses and check consistency
    """
    responses = []

    for i in range(n_samples):
        response = model.generate(
            query,
            temperature=0.7,  # Some randomness
            max_tokens=150
        )
        responses.append(response)

    # Check if all responses are similar
    consistency_score = calculate_semantic_similarity(responses)

    return {
        "responses": responses,
        "consistency_score": consistency_score,
        "likely_hallucination": consistency_score < 0.7  # Threshold
    }

def calculate_semantic_similarity(responses: list) -> float:
    """
    Calculate average pairwise similarity between responses
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(responses)

    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 0.0

# Usage
result = detect_hallucination_consistency(
    "What is the capital of France?",
    model
)

if result["likely_hallucination"]:
    print("⚠️ Potential hallucination detected - inconsistent responses")
```

#### **Contextual Consistency (RAG)**
```python
def check_context_consistency(query: str, context: str, response: str) -> bool:
    """
    Check if response is consistent with provided context
    """

    consistency_prompt = f"""
    Given the following context and response to a query, determine if the response is fully supported by the context.

    Context: {context}

    Query: {query}

    Response: {response}

    Is the response fully supported by the context? Consider:
    - Are all facts mentioned in the response present in the context?
    - Does the response make claims not supported by the context?
    - Is the response making logical inferences beyond what's stated?

    Answer: YES (fully supported) or NO (contains unsupported claims)
    """

    judge_response = llm.generate(consistency_prompt, temperature=0)

    return "YES" in judge_response.upper()

# Usage
is_consistent = check_context_consistency(
    query="How much does Product X cost?",
    context="Product X is available in our store. Product Y costs $50.",
    response="Product X costs $45."  # ❌ Not in context!
)
```

### 2. **Knowledge Verification**

#### **External Knowledge Base Lookup**
```python
import requests

def verify_factual_claim(claim: str) -> dict:
    """
    Verify factual claims against external knowledge sources
    """

    # Example: Wikidata API
    def query_wikidata(entity: str):
        # Simplified - real implementation would be more complex
        api_url = f"https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": entity,
            "language": "en",
            "format": "json"
        }
        response = requests.get(api_url, params=params)
        return response.json()

    # Extract entities from claim
    entities = extract_entities(claim)  # NER

    verifications = []
    for entity in entities:
        try:
            external_data = query_wikidata(entity)
            verification = compare_claim_with_external_data(claim, external_data)
            verifications.append(verification)
        except Exception as e:
            verifications.append({"entity": entity, "verified": False, "error": str(e)})

    return {
        "claim": claim,
        "verifications": verifications,
        "overall_confidence": calculate_overall_confidence(verifications)
    }

def extract_entities(text: str) -> list:
    """Extract named entities for verification"""
    from transformers import pipeline

    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    entities = ner(text)

    return [ent["word"] for ent in entities if ent["entity"] in ["B-PER", "B-ORG", "B-LOC"]]
```

#### **Citation Verification**
```python
import re

def verify_citations(response: str) -> dict:
    """
    Check if cited sources actually exist
    """

    # Extract citations (simplified patterns)
    citation_patterns = [
        r'([A-Z][a-z]+ et al\. \((\d{4})\))',  # "Smith et al. (2023)"
        r'([A-Z][a-z]+ \((\d{4})\))',          # "Smith (2023)"
        r'DOI: (10\.\d+\/[\w\.-]+)',           # DOI
        r'arXiv:(\d+\.\d+)',                   # arXiv
    ]

    found_citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, response)
        found_citations.extend(matches)

    # Verify each citation
    verified_citations = []
    for citation in found_citations:
        verification = verify_single_citation(citation)
        verified_citations.append(verification)

    return {
        "total_citations": len(found_citations),
        "verified_citations": verified_citations,
        "hallucinated_citations": [c for c in verified_citations if not c["exists"]]
    }

def verify_single_citation(citation: str) -> dict:
    """
    Verify if a single citation exists
    """
    # Example: Check arXiv, Google Scholar, etc.
    # This would involve API calls to academic databases

    return {
        "citation": citation,
        "exists": True,  # Placeholder - real implementation needed
        "confidence": 0.85,
        "source": "arXiv"
    }
```

### 3. **Confidence-Based Detection**

#### **Model Confidence Scores**
```python
import torch

def get_generation_confidence(model, tokenizer, prompt: str, response: str) -> float:
    """
    Calculate model's confidence in generated response
    """

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    response_tokens = tokenizer(response, return_tensors="pt")

    # Get logits
    with torch.no_grad():
        outputs = model(**inputs, labels=response_tokens["input_ids"])
        logits = outputs.logits

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get probability of actual tokens
    token_probs = []
    for i, token_id in enumerate(response_tokens["input_ids"][0]):
        if i < len(probs[0]):
            token_prob = probs[0][i][token_id].item()
            token_probs.append(token_prob)

    # Average confidence
    avg_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

    return avg_confidence

# Usage
confidence = get_generation_confidence(model, tokenizer, prompt, response)

if confidence < 0.5:  # Low confidence threshold
    print("⚠️ Low model confidence - potential hallucination")
```

#### **Uncertainty Quantification**
```python
def monte_carlo_dropout_uncertainty(model, prompt: str, n_samples: int = 10) -> dict:
    """
    Use MC Dropout to estimate uncertainty
    """
    model.train()  # Enable dropout

    responses = []
    confidences = []

    for _ in range(n_samples):
        response = model.generate(prompt, temperature=0.1)
        confidence = get_generation_confidence(model, tokenizer, prompt, response)

        responses.append(response)
        confidences.append(confidence)

    model.eval()  # Disable dropout

    # Calculate uncertainty metrics
    mean_confidence = sum(confidences) / len(confidences)
    std_confidence = np.std(confidences)

    return {
        "mean_confidence": mean_confidence,
        "uncertainty": std_confidence,
        "responses": responses,
        "high_uncertainty": std_confidence > 0.1  # Threshold
    }
```

### 4. **LLM-as-Judge Detection**

```python
def detect_hallucination_llm_judge(query: str, context: str, response: str) -> dict:
    """
    Use another LLM to judge if response contains hallucinations
    """

    judge_prompt = f"""
    You are an expert fact-checker. Analyze the following response for potential hallucinations.

    Query: {query}

    Context: {context}

    Response: {response}

    Check for:
    1. Factual accuracy - are the facts correct?
    2. Source attribution - are cited sources real?
    3. Context consistency - does the response stick to the provided context?
    4. Logical consistency - are there internal contradictions?

    Provide your analysis in this format:
    FACTUAL_ACCURACY: [PASS/FAIL] - [reason]
    SOURCE_ATTRIBUTION: [PASS/FAIL] - [reason]
    CONTEXT_CONSISTENCY: [PASS/FAIL] - [reason]
    LOGICAL_CONSISTENCY: [PASS/FAIL] - [reason]

    OVERALL: [PASS/FAIL]
    CONFIDENCE: [0.0-1.0]
    HALLUCINATION_RISK: [LOW/MEDIUM/HIGH]
    """

    judge_response = llm.generate(judge_prompt, temperature=0)

    # Parse response
    result = parse_judge_response(judge_response)

    return result

def parse_judge_response(response: str) -> dict:
    """Parse LLM judge response into structured format"""
    import re

    patterns = {
        "factual_accuracy": r"FACTUAL_ACCURACY: (\w+)",
        "source_attribution": r"SOURCE_ATTRIBUTION: (\w+)",
        "context_consistency": r"CONTEXT_CONSISTENCY: (\w+)",
        "logical_consistency": r"LOGICAL_CONSISTENCY: (\w+)",
        "overall": r"OVERALL: (\w+)",
        "confidence": r"CONFIDENCE: ([\d\.]+)",
        "risk": r"HALLUCINATION_RISK: (\w+)"
    }

    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, response)
        if match:
            results[key] = match.group(1)

    return results
```

## 🛡️ Prevention-Strategien

### 1. **Prompt Engineering**

#### **Explicit Instructions**
```python
def create_hallucination_resistant_prompt(query: str, context: str) -> str:
    """
    Create prompt that reduces hallucination risk
    """

    prompt = f"""
    You are a helpful assistant. IMPORTANT: Only answer based on the provided context.

    RULES:
    1. If the answer is not in the context, say "I don't have enough information to answer that."
    2. Never make up facts, dates, numbers, or quotes
    3. Never cite sources that aren't provided
    4. If you're uncertain, express that uncertainty
    5. Stick strictly to what's stated in the context

    Context: {context}

    Question: {query}

    Answer (based only on the context above):
    """

    return prompt
```

#### **Chain-of-Thought with Verification**
```python
def cot_with_verification_prompt(query: str, context: str) -> str:
    """
    CoT prompt that includes verification steps
    """

    prompt = f"""
    Context: {context}

    Question: {query}

    Please answer step by step:

    Step 1: Identify what information from the context is relevant to the question.
    Step 2: Check if the context contains enough information to answer the question.
    Step 3: If yes, provide the answer based only on the context. If no, state that you don't have enough information.
    Step 4: Verify that your answer doesn't contain any information not present in the context.

    Answer:
    """

    return prompt
```

### 2. **Retrieval Quality (RAG)**

#### **Relevance Filtering**
```python
def filter_irrelevant_chunks(query: str, chunks: list, threshold: float = 0.7) -> list:
    """
    Filter out chunks that aren't relevant enough
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_embedding = model.encode([query])
    chunk_embeddings = model.encode(chunks)

    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    relevant_chunks = [
        chunk for chunk, sim in zip(chunks, similarities)
        if sim >= threshold
    ]

    return relevant_chunks

# Usage
chunks = vector_db.query(query, top_k=10)
relevant_chunks = filter_irrelevant_chunks(query, chunks)

if not relevant_chunks:
    return "I don't have relevant information to answer your question."
```

#### **Multi-Vector Verification**
```python
def retrieve_with_verification(query: str, vector_db) -> dict:
    """
    Retrieve chunks and verify consistency
    """

    # Get multiple retrieval methods
    dense_results = vector_db.dense_search(query, top_k=5)
    sparse_results = vector_db.sparse_search(query, top_k=5)

    # Check consistency between methods
    overlap = set(dense_results) & set(sparse_results)

    return {
        "chunks": list(overlap) if overlap else dense_results[:3],
        "confidence": len(overlap) / 5,  # How many chunks agree
        "retrieval_quality": "high" if len(overlap) >= 3 else "low"
    }
```

### 3. **Generation Parameters**

```python
def generate_with_hallucination_prevention(prompt: str, model) -> str:
    """
    Generate with parameters that reduce hallucinations
    """

    response = model.generate(
        prompt,
        temperature=0.1,        # ✅ Lower creativity
        top_p=0.8,             # ✅ Focus on high-probability tokens
        top_k=40,              # ✅ Limit vocabulary
        repetition_penalty=1.1, # ✅ Avoid repetition
        max_tokens=200,        # ✅ Shorter responses
        do_sample=True
    )

    return response
```

### 4. **Post-Generation Filtering**

```python
def post_generation_filter(response: str, context: str, query: str) -> str:
    """
    Filter response after generation to remove hallucinations
    """

    # Check for hallucinations
    hallucination_check = detect_hallucination_llm_judge(query, context, response)

    if hallucination_check["overall"] == "FAIL":
        # Regenerate with stricter parameters
        stricter_prompt = create_hallucination_resistant_prompt(query, context)

        new_response = generate_with_hallucination_prevention(stricter_prompt, model)

        # Re-check
        second_check = detect_hallucination_llm_judge(query, context, new_response)

        if second_check["overall"] == "FAIL":
            return "I don't have reliable information to answer that question."

        return new_response

    return response
```

## 📊 Tools & Frameworks

### 1. **TruthfulQA Evaluation**
```python
# Evaluate model on TruthfulQA benchmark
from datasets import load_dataset

truthfulqa = load_dataset("truthful_qa", "generation")

def evaluate_truthfulness(model, n_samples: int = 100):
    """
    Evaluate model truthfulness on TruthfulQA
    """
    results = []

    for i, example in enumerate(truthfulqa["validation"]):
        if i >= n_samples:
            break

        question = example["question"]
        response = model.generate(question)

        # Human evaluation would be needed for full accuracy
        # This is a simplified version
        truthfulness_score = evaluate_response_truthfulness(response, example)

        results.append({
            "question": question,
            "response": response,
            "truthfulness": truthfulness_score
        })

    avg_truthfulness = sum(r["truthfulness"] for r in results) / len(results)

    return {
        "average_truthfulness": avg_truthfulness,
        "results": results
    }
```

### 2. **RAGAS Hallucination Metrics**
```python
from ragas import evaluate
from ragas.metrics import faithfulness

def evaluate_rag_hallucinations(questions: list, contexts: list, responses: list):
    """
    Use RAGAS to evaluate RAG hallucinations
    """

    dataset = {
        "question": questions,
        "contexts": contexts,
        "answer": responses
    }

    # Faithfulness measures hallucinations
    score = evaluate(dataset, metrics=[faithfulness])

    return score
```

### 3. **Production Monitoring**
```python
class HallucinationMonitor:
    def __init__(self):
        self.detection_methods = [
            self.consistency_check,
            self.confidence_check,
            self.context_check
        ]

    def monitor_response(self, query: str, context: str, response: str) -> dict:
        """
        Monitor response for hallucinations
        """

        results = {}

        for method in self.detection_methods:
            try:
                result = method(query, context, response)
                results[method.__name__] = result
            except Exception as e:
                results[method.__name__] = {"error": str(e)}

        # Aggregate results
        hallucination_detected = any(
            r.get("hallucination_detected", False)
            for r in results.values()
            if not r.get("error")
        )

        return {
            "hallucination_detected": hallucination_detected,
            "method_results": results,
            "confidence": self.calculate_aggregate_confidence(results)
        }

    def consistency_check(self, query: str, context: str, response: str) -> dict:
        # Implementation of consistency checking
        pass

    def confidence_check(self, query: str, context: str, response: str) -> dict:
        # Implementation of confidence checking
        pass

    def context_check(self, query: str, context: str, response: str) -> dict:
        # Implementation of context checking
        pass

# Usage
monitor = HallucinationMonitor()

result = monitor.monitor_response(query, context, response)

if result["hallucination_detected"]:
    log_security_event("hallucination_detected", result)
```

## 🎓 Weiterführende Themen

### Verwandte Kapitel
- **[../training/05-regularization.md](../training/05-regularization.md)** - Training-Techniken gegen Overfitting
- **[../../05-ethics/02-transparency/](../../05-ethics/02-transparency/)** - Transparency & Explainability
- **[../../04-advanced/02-prompt-engineering.md](../../04-advanced/02-prompt-engineering.md)** - Advanced Prompting Techniques

### Nächste Schritte
1. **Implementiere Monitoring**: Production Hallucination Detection
2. **Evaluation Setup**: TruthfulQA, RAGAS Benchmarks
3. **Prevention Integration**: In RAG-Pipeline einbauen

## 📚 Ressourcen

### Papers
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)
- [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629)
- [Faithful or Extractive? On Mitigating the Faithfulness-Abstractiveness Trade-off](https://arxiv.org/abs/2108.13684)

### Tools & Libraries
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG Evaluation Framework
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) - Benchmark für Truthfulness
- [FActScore](https://github.com/shmsw25/FActScore) - Fine-grained Factuality Evaluation

### Blog Posts
- [Reducing Hallucinations in LLMs](https://platform.openai.com/docs/guides/reducing-hallucinations)
- [RAG vs Hallucinations](https://blog.langchain.dev/rag-vs-hallucinations/)