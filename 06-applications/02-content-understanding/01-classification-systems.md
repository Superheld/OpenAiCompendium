# Classification Systems: Automatische Kategorisierung

## 🎯 Ziel
Entwickle intelligente Klassifikationssysteme von traditionellen ML-Ansätzen bis zu modernen Transformer-basierten Lösungen - für automatische Content-Kategorisierung, Sentiment-Analyse und Document Processing.

## 📖 Geschichte & Kontext

**Das Problem:** Massive Mengen unstrukturierter Daten müssen automatisch kategorisiert werden.

**Evolution der Classification:**
- **1990er**: Naive Bayes, SVM mit handgefertigten Features
- **2000er**: Random Forests, Gradient Boosting
- **2010er**: Deep Learning (CNNs für Text, RNNs)
- **2020er**: Transformer-based Classification (BERT, RoBERTa)

**Warum automatische Classification wichtig ist:**
- Manuelle Kategorisierung skaliert nicht
- Konsistente, objektive Klassifikation
- Real-time Processing von Content
- Basis für weitere AI-Systeme (Search, Recommendation)

**Typische Anwendungen:**
- **Content Moderation**: Spam, Hate Speech, NSFW Detection
- **Document Classification**: Legal, Medical, Technical Documents
- **Sentiment Analysis**: Customer Reviews, Social Media
- **Intent Classification**: Chatbots, Customer Service

## 🧮 Konzept & Theorie

### Classification System Typen

**1. Binary Classification**
```python
Input: "Dieses Produkt ist fantastisch!"
Output: POSITIVE (vs. NEGATIVE)
```

**2. Multi-Class Classification**
```python
Input: "Laborkühlschrank HMFvh 4001 Bedienungsanleitung"
Output: TECHNICAL_DOCUMENTATION (vs. MARKETING, LEGAL, ...)
```

**3. Multi-Label Classification**
```python
Input: "Medikamentenkühlschrank für Apotheken"
Output: [MEDICAL, EQUIPMENT, STORAGE, PHARMACY]
```

**4. Hierarchical Classification**
```python
Input: "Impfstoff-Kühlschrank"
Output: Medical > Equipment > Storage > Refrigeration
```

### Die Classification Pipeline

**Phase 1: Data Preparation**
```
Raw Text → Cleaning → Tokenization → Feature Extraction → Training Data
```

**Phase 2: Model Training**
```
Features + Labels → Model Training → Validation → Hyperparameter Tuning → Final Model
```

**Phase 3: Inference**
```
New Text → Feature Extraction → Model Prediction → Confidence Score → Classification
```

## 🛠️ Implementation

### 1. Classical ML Classification

**Warum Classical ML?**
- Interpretierbar und erklärbar
- Wenig Trainingsdaten benötigt
- Schnell zu trainieren
- Baseline für komplexere Ansätze

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Beispiel-Daten: Produktkategorisierung
data = {
    'text': [
        "Laborkühlschrank mit 8 Schubfädern für Medikamente",
        "High-End Gaming Laptop mit RTX 4090 Grafikkarte",
        "Antibiotika Amoxicillin 500mg Tabletten",
        "Bürostuhl ergonomisch mit Lordosenstütze",
        "Blutdruckmessgerät digital automatisch",
        "Smartphone 5G mit 128GB Speicher",
        "Insulin Pen für Diabetiker",
        "Labortisch höhenverstellbar Edelstahl",
        "Tablet 10 Zoll Android",
        "Desinfektionsmittel 70% Alkohol",
        # ... mehr Trainingsdaten
    ],
    'category': [
        'MEDICAL_EQUIPMENT', 'ELECTRONICS', 'MEDICINE', 'FURNITURE',
        'MEDICAL_EQUIPMENT', 'ELECTRONICS', 'MEDICINE', 'FURNITURE',
        'ELECTRONICS', 'MEDICINE'
        # ... entsprechende Labels
    ]
}

df = pd.DataFrame(data)

# Text Preprocessing
stop_words = set(stopwords.words('german'))

def preprocess_text(text):
    # Tokenisierung
    tokens = word_tokenize(text.lower(), language='german')
    # Stopwords entfernen
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

# Feature Extraction mit TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams und Bigrams
    min_df=2,  # Mindestens 2 Dokumente
    max_df=0.95  # Maximal 95% der Dokumente
)

X = vectorizer.fit_transform(df['processed_text'])
y = df['category']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model Training
models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    # Training
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))

# Feature Importance Analysis (für Logistic Regression)
lr_model = results['Logistic Regression']['model']
feature_names = vectorizer.get_feature_names_out()

# Top Features pro Kategorie
for i, category in enumerate(lr_model.classes_):
    coefficients = lr_model.coef_[i]
    top_indices = np.argsort(coefficients)[-10:]  # Top 10 Features

    print(f"\nTop Features für {category}:")
    for idx in reversed(top_indices):
        print(f"  {feature_names[idx]}: {coefficients[idx]:.3f}")

# Prediction Function
def predict_category(text, model_name='Logistic Regression'):
    processed = preprocess_text(text)
    features = vectorizer.transform([processed])
    model = results[model_name]['model']

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Top 3 Predictions mit Confidence
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_predictions = [
        (model.classes_[i], probabilities[i])
        for i in top_indices
    ]

    return prediction, top_predictions

# Test Predictions
test_texts = [
    "Stethoskop für Herzuntersuchung",
    "MacBook Pro M3 für Entwickler",
    "Aspirin 100mg Herzschutz",
    "Schreibtisch Eiche massiv"
]

for text in test_texts:
    pred, top_preds = predict_category(text)
    print(f"\nText: {text}")
    print(f"Prediction: {pred}")
    print("Top 3 Confidences:")
    for category, conf in top_preds:
        print(f"  {category}: {conf:.3f}")
```

### 2. Modern Deep Learning Classification

**Warum Deep Learning?**
- Bessere Performance bei komplexen Patterns
- Transfer Learning möglich
- Weniger Feature Engineering
- State-of-the-art Ergebnisse

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import datasets

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class ModernTextClassifier:
    def __init__(self, model_name='distilbert-base-german-cased', num_labels=4):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.label_encoder = LabelEncoder()

    def prepare_data(self, texts, labels):
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42
        )

        # Create datasets
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer
        )
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, self.tokenizer
        )

        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset, output_dir='./results'):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        return trainer

    def predict(self, texts):
        self.model.eval()
        predictions = []

        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )

            with torch.no_grad():
                outputs = self.model(**encoding)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()

            # Decode prediction
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = probabilities[0][predicted_class].item()

            predictions.append({
                'text': text,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'all_probabilities': {
                    self.label_encoder.inverse_transform([i])[0]: prob.item()
                    for i, prob in enumerate(probabilities[0])
                }
            })

        return predictions

# Usage Example
classifier = ModernTextClassifier(
    model_name='distilbert-base-german-cased',
    num_labels=len(df['category'].unique())
)

# Prepare data
train_dataset, val_dataset = classifier.prepare_data(
    df['text'].tolist(),
    df['category'].tolist()
)

# Train model
trainer = classifier.train(train_dataset, val_dataset)

# Test predictions
test_texts = [
    "Digitales Thermometer für Fiebermessung",
    "Gaming Maus mit RGB Beleuchtung",
    "Vitamin D3 Kapseln hochdosiert",
    "Konferenztisch für 12 Personen"
]

predictions = classifier.predict(test_texts)

for pred in predictions:
    print(f"\nText: {pred['text']}")
    print(f"Prediction: {pred['predicted_label']} (Confidence: {pred['confidence']:.3f})")
    print("All Probabilities:")
    for label, prob in sorted(pred['all_probabilities'].items(),
                            key=lambda x: x[1], reverse=True):
        print(f"  {label}: {prob:.3f}")
```

### 3. Production Classification System

```python
import mlflow
import pickle
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging

class ProductionClassificationSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.performance_metrics = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train ensemble of different models"""

        # Classical Models
        classical_models = {
            'nb': MultinomialNB(alpha=1.0),
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }

        for name, model in classical_models.items():
            self.logger.info(f"Training {name}...")

            # Train
            model.fit(X_train, y_train)

            # Validate
            val_pred = model.predict(X_val)
            val_prob = model.predict_proba(X_val)

            # Store metrics
            self.performance_metrics[name] = {
                'accuracy': accuracy_score(y_val, val_pred),
                'f1_macro': f1_score(y_val, val_pred, average='macro'),
                'classification_report': classification_report(y_val, val_pred, output_dict=True)
            }

            # Store model
            self.models[name] = model

    def predict_ensemble(self, texts: List[str], return_probabilities=True):
        """Ensemble prediction with confidence scoring"""

        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        features = self.vectorizers['tfidf'].transform(processed_texts)

        ensemble_predictions = []

        for text_idx, text in enumerate(texts):
            text_features = features[text_idx:text_idx+1]
            model_predictions = {}
            model_probabilities = {}

            # Get predictions from all models
            for model_name, model in self.models.items():
                pred = model.predict(text_features)[0]
                prob = model.predict_proba(text_features)[0]

                model_predictions[model_name] = pred
                model_probabilities[model_name] = prob

            # Ensemble voting (weighted by performance)
            ensemble_prob = np.zeros(len(self.models['lr'].classes_))

            for model_name, prob in model_probabilities.items():
                weight = self.performance_metrics[model_name]['f1_macro']
                ensemble_prob += weight * prob

            ensemble_prob /= len(self.models)

            # Final prediction
            final_prediction = self.models['lr'].classes_[np.argmax(ensemble_prob)]
            confidence = np.max(ensemble_prob)

            result = {
                'text': text,
                'prediction': final_prediction,
                'confidence': confidence,
                'individual_predictions': model_predictions
            }

            if return_probabilities:
                result['class_probabilities'] = {
                    class_name: prob
                    for class_name, prob in zip(self.models['lr'].classes_, ensemble_prob)
                }

            ensemble_predictions.append(result)

        return ensemble_predictions

    def model_monitoring(self, predictions: List[Dict], feedback: List[str] = None):
        """Monitor model performance and drift"""

        # Confidence distribution
        confidences = [pred['confidence'] for pred in predictions]

        monitoring_report = {
            'timestamp': datetime.now().isoformat(),
            'num_predictions': len(predictions),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'low_confidence_ratio': sum(1 for c in confidences if c < 0.7) / len(confidences),
            'class_distribution': {}
        }

        # Class distribution
        predicted_classes = [pred['prediction'] for pred in predictions]
        unique_classes, counts = np.unique(predicted_classes, return_counts=True)

        for class_name, count in zip(unique_classes, counts):
            monitoring_report['class_distribution'][class_name] = count

        # If feedback available, calculate accuracy
        if feedback:
            correct_predictions = sum(
                1 for pred, true_label in zip(predicted_classes, feedback)
                if pred == true_label
            )
            monitoring_report['accuracy'] = correct_predictions / len(feedback)

        return monitoring_report

    def save_model(self, path: str):
        """Save complete model pipeline"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'label_encoders': self.label_encoders,
            'performance_metrics': self.performance_metrics,
            'config': self.config
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str):
        """Load complete model pipeline"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(model_data['config'])
        instance.models = model_data['models']
        instance.vectorizers = model_data['vectorizers']
        instance.label_encoders = model_data['label_encoders']
        instance.performance_metrics = model_data['performance_metrics']

        return instance

# Usage Example
config = {
    'max_features': 10000,
    'ngram_range': (1, 2),
    'confidence_threshold': 0.7
}

# Initialize system
classification_system = ProductionClassificationSystem(config)

# Train ensemble (using previous data)
# classification_system.train_ensemble(X_train, y_train, X_val, y_val)

# Make predictions
test_texts = [
    "Ultraschallgerät für medizinische Diagnostik",
    "4K Monitor für Videobearbeitung",
    "Ibuprofen 400mg Schmerztabletten",
    "Chefsessel Leder schwarz"
]

# predictions = classification_system.predict_ensemble(test_texts)

# Monitor performance
# monitoring_report = classification_system.model_monitoring(predictions)
```

## 🛠️ Tools & Frameworks

### **Classical ML**
- **scikit-learn** - Comprehensive ML library with all classical algorithms
- **XGBoost/LightGBM** - Gradient boosting frameworks for tabular data
- **spaCy** - Industrial-strength NLP with classification capabilities
- **nltk** - Natural language processing toolkit

### **Deep Learning**
- **Transformers** (Hugging Face) - State-of-the-art transformer models
- **PyTorch/TensorFlow** - Deep learning frameworks
- **Lightning** - High-level PyTorch wrapper
- **Optuna** - Hyperparameter optimization

### **Production & MLOps**
- **MLflow** - ML lifecycle management
- **Weights & Biases** - Experiment tracking
- **BentoML** - Model serving framework
- **Seldon Core** - Kubernetes-native ML deployment

### **Pre-trained Models**
- **bert-base-german-cased** - German BERT model
- **distilbert-base-german-cased** - Faster German BERT
- **gbert-large** - Large German BERT model
- **xlm-roberta-large** - Multilingual RoBERTa

## 🚀 Was du danach kannst

**Grundlagen:**
- Du verstehst verschiedene Classification-Paradigmen (Binary, Multi-Class, Multi-Label)
- Du implementierst Classical ML und Deep Learning Classifier
- Du evaluierst Classification-Performance mit geeigneten Metriken

**Production-Skills:**
- Du entwickelst robuste Ensemble-Classifier für bessere Performance
- Du implementierst Model Monitoring und Drift Detection
- Du optimierst Classification-Systeme für verschiedene Business-Metriken

**Advanced:**
- Du kennst moderne Transfer Learning Techniken für wenig Trainingsdaten
- Du implementierst Few-Shot und Zero-Shot Classification
- Du integrierst Classification in komplexere AI-Pipelines

## 🔗 Weiterführende Themen
- **Search Systems**: [02-search-systems.md](02-search-systems.md) für classification-basierte Suche
- **Evaluation**: [../03-core/evaluation/04-quality-metrics.md](../03-core/evaluation/04-quality-metrics.md) für Classification-Metriken
- **Ethics**: [../05-ethics/01-bias-fairness/](../05-ethics/01-bias-fairness/) für faire Klassifikation
- **Training**: [../03-core/training/](../03-core/training/) für Fine-Tuning Techniken