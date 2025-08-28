# Production Email Classifier API Documentation

**Version**: 1.0.0  
**Accuracy**: 98.18% on real-world data  
**Last Updated**: August 27, 2025

## Overview

The Production Email Classifier provides a simple, robust API for classifying emails as job-related or not. It uses a pre-trained XGBoost model with 575 generalized features to achieve 98%+ accuracy.

## Quick Start

```python
from src.models.production_email_classifier import ProductionEmailClassifier

# Initialize and load model
classifier = ProductionEmailClassifier()
classifier.load_model()

# Classify single email
result = classifier.classify_email(
    email_body="Thank you for applying to our Software Engineer position.",
    sender_email="hr@company.com"
)
print(result)
```

## API Reference

### Class: ProductionEmailClassifier

#### `__init__(model_dir=None)`
Initialize the classifier.

**Parameters:**
- `model_dir` (str, optional): Directory containing model files. Defaults to `data/models/`

**Example:**
```python
# Use default model directory
classifier = ProductionEmailClassifier()

# Use custom model directory  
classifier = ProductionEmailClassifier("/path/to/models")
```

#### `load_model()`
Load the trained model and feature pipeline.

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**
```python
if classifier.load_model():
    print("Model loaded successfully")
else:
    print("Failed to load model")
```

**Required Files:**
- `generalized_email_classifier.pkl`
- `generalized_feature_pipeline.pkl`

#### `classify_email(email_body, sender_email=None, date=None)`
Classify a single email.

**Parameters:**
- `email_body` (str): Email content to classify
- `sender_email` (str, optional): Sender's email address
- `date` (str, optional): Email date in various formats

**Returns:**
- `dict`: Classification result

**Response Format:**
```python
{
    'is_job_related': True,              # Boolean result
    'prediction': 'job_related',         # String prediction
    'probability': 0.9234,               # Probability score (0-1)
    'confidence': 'high',                # 'high', 'medium', or 'low'
    'model_version': '1.0.0',           # Model version
    'processed_at': '2025-08-27T10:30:00' # Timestamp
}
```

**Example:**
```python
result = classifier.classify_email(
    email_body="We would like to schedule an interview with you.",
    sender_email="recruiter@techcorp.com",
    date="2025-08-27"
)

if result['is_job_related']:
    print(f"Job email detected with {result['probability']:.1%} confidence")
```

#### `classify_batch(emails)`
Classify multiple emails at once.

**Parameters:**
- `emails` (list): List of email dictionaries

**Email Dictionary Format:**
```python
{
    'email_body': str,      # Required
    'sender_email': str,    # Optional
    'date': str            # Optional
}
```

**Returns:**
- `list`: List of classification results with `batch_index` added

**Example:**
```python
emails = [
    {
        'email_body': "Thank you for your application...",
        'sender_email': "hr@company.com"
    },
    {
        'email_body': "Your order has been shipped...",
        'sender_email': "orders@amazon.com"
    }
]

results = classifier.classify_batch(emails)
for result in results:
    print(f"Email {result['batch_index']}: {result['prediction']}")
```

#### `get_model_info()`
Get information about the loaded model.

**Returns:**
- `dict`: Model metadata

**Response Format:**
```python
{
    'model_name': 'Generalized Email Classifier',
    'model_version': '1.0.0',
    'accuracy': 0.98,
    'feature_count': 575,
    'is_loaded': True,
    'model_type': 'XGBoost Classifier',
    'training_date': '2025-08-27',
    'generalized': True
}
```

#### `health_check()`
Perform a health check on the classifier.

**Returns:**
- `dict`: Health status

**Response Format:**
```python
{
    'status': 'healthy',           # 'healthy' or 'unhealthy'
    'model_version': '1.0.0',
    'test_prediction': 'job_related',
    'timestamp': '2025-08-27T10:30:00'
}
```

## Confidence Levels

The classifier returns confidence levels based on prediction probability:

| Confidence | Probability Range | Meaning |
|------------|------------------|---------|
| **high** | > 0.9 or < 0.1 | Very confident in prediction |
| **medium** | 0.7-0.9 or 0.1-0.3 | Moderately confident |
| **low** | 0.3-0.7 | Less confident, manual review recommended |

## Error Handling

All methods return error information in case of failure:

```python
result = classifier.classify_email("invalid input")
if result.get('error'):
    print(f"Error: {result['error']}")
    # Handle error appropriately
```

**Common Errors:**
- Model not loaded: Call `load_model()` first
- Invalid input: Provide non-empty email_body
- Missing files: Ensure model files exist

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 98.18% | Validated on external dataset |
| **Processing Speed** | ~0.001s/email | Single email classification |
| **Memory Usage** | ~50MB | Model + feature pipeline |
| **Batch Efficiency** | Linear scaling | No significant overhead |

## Integration Examples

### Simple Classification Service
```python
class EmailService:
    def __init__(self):
        self.classifier = ProductionEmailClassifier()
        self.classifier.load_model()
    
    def is_job_email(self, email_content, sender=None):
        result = self.classifier.classify_email(email_content, sender)
        return result.get('is_job_related', False)
```

### Email Filter with Confidence Threshold
```python
def filter_job_emails(emails, confidence_threshold='medium'):
    classifier = ProductionEmailClassifier()
    classifier.load_model()
    
    job_emails = []
    for email in emails:
        result = classifier.classify_email(email['body'], email.get('sender'))
        
        if result['is_job_related'] and result['confidence'] != 'low':
            job_emails.append({
                'email': email,
                'probability': result['probability'],
                'confidence': result['confidence']
            })
    
    return job_emails
```

### Batch Processing with Error Handling
```python
def process_email_batch(email_data):
    classifier = ProductionEmailClassifier()
    if not classifier.load_model():
        return {'error': 'Failed to load model'}
    
    results = classifier.classify_batch(email_data)
    
    successful = [r for r in results if not r.get('error')]
    failed = [r for r in results if r.get('error')]
    
    return {
        'total': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'job_emails': [r for r in successful if r['is_job_related']],
        'errors': failed
    }
```

## Production Deployment

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify model files exist
ls data/models/generalized_email_classifier.pkl
ls data/models/generalized_feature_pipeline.pkl
```

### Health Check Endpoint
```python
from flask import Flask, jsonify
app = Flask(__name__)

classifier = ProductionEmailClassifier()
classifier.load_model()

@app.route('/health')
def health():
    return jsonify(classifier.health_check())

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    result = classifier.classify_email(
        data['email_body'],
        data.get('sender_email')
    )
    return jsonify(result)
```

### Monitoring Recommendations

1. **Track Key Metrics:**
   - Classification accuracy over time
   - Confidence score distribution
   - Processing latency
   - Error rates

2. **Set Up Alerts:**
   - Low confidence predictions > 5%
   - Processing time > 100ms
   - Error rate > 1%

3. **Log Important Events:**
   - Model loading/reloading
   - Classification errors
   - Unusual confidence patterns

## Support & Troubleshooting

### Common Issues

1. **Model files not found**
   ```
   Solution: Ensure generalized_email_classifier.pkl and 
   generalized_feature_pipeline.pkl exist in data/models/
   ```

2. **Low accuracy in production**
   ```
   Solution: Check input data format matches training data.
   Monitor for distribution drift.
   ```

3. **Memory issues**
   ```
   Solution: Load model once at startup, reuse classifier instance.
   Consider batch processing for high volume.
   ```

### Model Updates

The model should be retrained when:
- Accuracy drops below 95% in production
- New email patterns emerge (quarterly review)
- User feedback indicates systematic errors

### Version Compatibility

- **Python**: 3.8+
- **Scikit-learn**: 1.0+
- **XGBoost**: 1.5+
- **Pandas**: 1.3+

---

**Contact**: For integration support, check the health endpoint and validate input data format.
**Model Version**: 1.0.0 (Production-Ready)
**Training Date**: August 27, 2025