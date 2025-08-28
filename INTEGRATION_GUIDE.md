# Production Integration Guide

## Email Classification System Integration

**Repository**: `https://github.com/zichengalexzhao/onlyjobs-classfication.git`  
**Branch**: `feature/new-data-format`  
**Model Version**: 1.0.0

---

## Quick Integration Steps

### 1. Clone and Install

```bash
git clone https://github.com/zichengalexzhao/onlyjobs-classfication.git
cd onlyjobs-classfication
git checkout feature/new-data-format
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from src.models.production_email_classifier import ProductionEmailClassifier

# Initialize and load model
classifier = ProductionEmailClassifier()
if not classifier.load_model():
    raise RuntimeError("Failed to load model files")

# Classify single email
result = classifier.classify_email(
    email_body="Thank you for applying to our Software Engineer position. We would like to schedule an interview.",
    sender_email="hr@company.com"
)

print(f"Job-related: {result['is_job_related']}")
print(f"Confidence: {result['confidence']}")
print(f"Probability: {result['probability']:.1%}")
```

### 3. Integration Patterns

#### A. Service Integration
```python
class EmailClassificationService:
    def __init__(self):
        self.classifier = ProductionEmailClassifier()
        if not self.classifier.load_model():
            raise RuntimeError("Model loading failed")
    
    def classify_emails(self, emails):
        """Classify batch of emails for service integration."""
        return self.classifier.classify_batch(emails)
    
    def health_check(self):
        """Health check for monitoring."""
        return self.classifier.health_check()
```

#### B. API Endpoint Integration
```python
from flask import Flask, jsonify, request

app = Flask(__name__)
classifier = ProductionEmailClassifier()
classifier.load_model()

@app.route('/classify', methods=['POST'])
def classify_email():
    data = request.get_json()
    result = classifier.classify_email(
        email_body=data.get('email_body'),
        sender_email=data.get('sender_email'),
        date=data.get('date')
    )
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify(classifier.health_check())
```

#### C. Background Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncEmailClassifier:
    def __init__(self):
        self.classifier = ProductionEmailClassifier()
        self.classifier.load_model()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def classify_async(self, email_body, sender_email=None):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.classifier.classify_email,
            email_body,
            sender_email
        )
```

---

## Production Performance

### Validated Metrics
- **External Dataset Accuracy**: 98.18% (494 real job emails)
- **Internal Test Accuracy**: 98.0% with 97.9% F1-score  
- **AUC Score**: 99.9% (near-perfect discrimination)
- **High Confidence Predictions**: 97% of all classifications
- **Processing Speed**: ~0.001s per email

### Error Patterns (1.82% error rate)
- 67% automated HR system notifications (very short content)
- 22% platform-specific formatting issues (ICIMS, Workday)
- 11% rejection emails misclassified

---

## Model Files Required

Ensure these files exist in your deployment:

```
data/models/
├── generalized_email_classifier.pkl      # XGBoost trained model
├── generalized_feature_pipeline.pkl      # Feature extraction pipeline
├── generalized_training_results.json     # Performance metrics
└── generalized_feature_analysis.json     # Feature importance data
```

**File Sizes**:
- Model: ~2.3MB
- Pipeline: ~1.1MB  
- Total: ~3.4MB

---

## Response Format

### Successful Classification
```json
{
    "is_job_related": true,
    "prediction": "job_related",
    "probability": 0.9234,
    "confidence": "high",
    "model_version": "1.0.0",
    "processed_at": "2025-08-28T10:30:00"
}
```

### Error Response
```json
{
    "error": "Classification failed: invalid input format",
    "is_job_related": null,
    "prediction": null,
    "probability": null,
    "confidence": null
}
```

### Confidence Levels
- **High**: >90% or <10% probability (97% of predictions)
- **Medium**: 70-90% or 10-30% probability (3% of predictions)
- **Low**: 30-70% probability (0% of predictions)

---

## Environment Setup

### Dependencies
```txt
pandas>=1.5.0
scikit-learn>=1.2.0
xgboost>=1.6.0
numpy>=1.21.0
```

### Python Version
- **Minimum**: Python 3.8+
- **Recommended**: Python 3.10+
- **Tested**: Python 3.11, 3.12, 3.13

### Memory Requirements
- **Model Loading**: ~50MB RAM
- **Per Classification**: ~1MB RAM
- **Batch Processing**: ~10MB RAM per 100 emails

---

## Integration Checklist

### Pre-deployment
- [ ] Clone repository and checkout `feature/new-data-format` branch
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify model files exist in `data/models/` directory
- [ ] Test basic classification: `python example_integration.py`
- [ ] Run health check: `classifier.health_check()`

### Production Setup
- [ ] Configure error logging and monitoring
- [ ] Set up model file backup and versioning
- [ ] Implement batch processing for high volumes
- [ ] Add performance monitoring (latency, accuracy)
- [ ] Configure health check endpoints

### Monitoring
- [ ] Track classification accuracy over time
- [ ] Monitor confidence score distribution
- [ ] Alert on high volume of low-confidence predictions
- [ ] Log processing times and system performance

---

## Advanced Features

### Batch Processing
```python
emails = [
    {"email_body": "Interview invitation...", "sender_email": "hr@tech.com"},
    {"email_body": "Order confirmation...", "sender_email": "orders@store.com"},
    # ... more emails
]

results = classifier.classify_batch(emails)
job_emails = [r for r in results if r.get('is_job_related')]
```

### Error Handling
```python
result = classifier.classify_email(email_text)

if result.get('error'):
    logger.error(f"Classification failed: {result['error']}")
    # Handle error appropriately
elif result['confidence'] == 'low':
    logger.warning(f"Low confidence prediction: {result['probability']:.1%}")
    # Consider manual review
```

### Performance Optimization
```python
# Load model once at application startup
classifier = ProductionEmailClassifier()
classifier.load_model()

# Reuse classifier instance for multiple predictions
# Thread-safe for read operations
```

---

## Troubleshooting

### Common Issues

#### Model Loading Fails
```python
# Check file paths
import os
model_dir = "data/models"
required_files = [
    "generalized_email_classifier.pkl",
    "generalized_feature_pipeline.pkl"
]

for file in required_files:
    path = os.path.join(model_dir, file)
    if not os.path.exists(path):
        print(f"Missing: {path}")
```

#### Low Performance
- Ensure you're using the correct branch: `feature/new-data-format`
- Verify model files are not corrupted
- Check Python environment has all dependencies

#### Integration Issues
- Review the `example_integration.py` for complete working examples
- Test with the command-line tool: `python scripts/classify_email.py`
- Check logs for detailed error messages

---

## Support and Updates

### Model Information
- **Training Date**: August 27, 2025
- **Training Data**: 2,000 generalized emails
- **Feature Count**: 575 user-agnostic features
- **Model Type**: XGBoost Classifier

### Future Updates
- Monitor for model drift in production
- Retrain quarterly with new email patterns
- Update feature engineering for new platforms
- Maintain backward compatibility

### Version Control
Current model outputs `model_version: "1.0.0"` in all responses for tracking.

---

**Production-Ready Email Classification**  
*98%+ accuracy validated on real-world data*