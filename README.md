# OnlyJobs Classification

**High-performance email classification system for identifying job-related emails using machine learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

## 🎯 Overview

OnlyJobs Classification is a machine learning system that accurately identifies job-related emails from regular email content. It provides:

- **90%+ accuracy** with Random Forest classification
- **2000x faster** than LLM-based approaches (0.001s vs 2s per email)
- **100% cost savings** (no API calls required)
- **Offline capability** with no external dependencies

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/onlyjobs-classification.git
cd onlyjobs-classification

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings

# Collect training data
python scripts/collect_data.py --target-samples 2000

# Train models
python scripts/train_models.py

# Run classification
python scripts/classify_email.py --email "Your email content here"
```

## 📁 Project Structure

```
onlyjobs-classification/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation
├── LICENSE                  # MIT License
│
├── config/                  # Configuration files
│   ├── config.yaml         # Main configuration
│   ├── gmail_credentials.json.example
│   └── .env.example
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_collection/    # Email data collection
│   │   ├── __init__.py
│   │   ├── gmail_collector.py
│   │   └── data_processor.py
│   │
│   ├── feature_engineering/ # Feature extraction
│   │   ├── __init__.py
│   │   ├── text_features.py
│   │   └── feature_pipeline.py
│   │
│   ├── models/             # ML models
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── model_utils.py
│   │
│   ├── evaluation/         # Model evaluation
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   └── deployment/         # Production deployment
│       ├── __init__.py
│       ├── api.py
│       └── batch_processor.py
│
├── scripts/                # Executable scripts
│   ├── collect_data.py     # Data collection script
│   ├── train_models.py     # Model training script
│   ├── evaluate_models.py  # Model evaluation
│   └── classify_email.py   # Single email classification
│
├── data/                   # Data storage
│   ├── raw/               # Raw email data
│   ├── processed/         # Processed features
│   ├── models/           # Trained models
│   └── results/          # Evaluation results
│
├── tests/                 # Unit tests
│   ├── __init__.py
│   ├── test_data_collection.py
│   ├── test_features.py
│   └── test_models.py
│
├── docs/                  # Documentation
│   ├── api_reference.md   # API documentation
│   ├── data_collection.md # Data collection guide
│   ├── model_training.md  # Training guide
│   └── deployment.md     # Deployment guide
│
└── examples/              # Usage examples
    ├── basic_usage.py     # Basic classification example
    ├── batch_processing.py # Batch processing example
    └── api_integration.py  # API integration example
```

## 🧠 Model Performance

Based on comprehensive cross-validation analysis:

| Model | F1-Score | Accuracy | Speed | Memory |
|-------|----------|----------|-------|---------|
| **Random Forest** ⭐ | **90.15%** | **90.00%** | 0.001s | 50MB |
| Gradient Boosting | 89.28% | 89.11% | 0.002s | 30MB |
| SVM | 89.28% | 89.17% | 0.001s | 40MB |
| Logistic Regression | 86.78% | 86.61% | 0.001s | 10MB |

## 📊 Features

### Data Collection
- **Gmail API integration** for real email data
- **Cost-efficient sampling** with budget controls
- **Balanced dataset creation** (positive/negative examples)
- **Privacy-conscious** processing

### Feature Engineering
- **1000+ TF-IDF features** from email content
- **Keyword-based features** (job-specific terms)
- **Structural features** (HTML, formatting, links)
- **Statistical features** (length, word count, etc.)

### Model Training
- **Multiple algorithms** (Random Forest, SVM, Gradient Boosting)
- **Cross-validation** with statistical significance testing
- **Hyperparameter optimization** for best performance
- **Overfitting detection** and prevention

### Production Deployment
- **REST API** for real-time classification
- **Batch processing** for high-volume scenarios
- **Model monitoring** and drift detection
- **A/B testing** framework

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- Gmail API credentials (for data collection)
- 2GB+ RAM for model training

### Install from Source
```bash
git clone https://github.com/yourusername/onlyjobs-classification.git
cd onlyjobs-classification
pip install -e .
```

### Install from PyPI
```bash
pip install onlyjobs-classification
```

## 📖 Usage

### Command Line Interface
```bash
# Collect training data
onlyjobs collect --samples 2000 --budget 5.00

# Train models
onlyjobs train --algorithm random_forest

# Classify single email
onlyjobs classify --text "Thank you for your application..."

# Batch process emails
onlyjobs batch --input emails.json --output results.json
```

### Python API
```python
from onlyjobs import EmailClassifier

# Initialize classifier
classifier = EmailClassifier()

# Load pre-trained model
classifier.load_model('path/to/model.pkl')

# Classify email
result = classifier.predict("Email content here")
print(f"Job-related: {result['is_job_related']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 🔬 Model Development

### Data Collection
```bash
# Set up Gmail API credentials
cp config/gmail_credentials.json.example config/gmail_credentials.json
# Edit with your credentials

# Collect balanced dataset
python scripts/collect_data.py \
    --positive-samples 1000 \
    --negative-samples 1000 \
    --budget 5.00
```

### Model Training
```bash
# Train all models with cross-validation
python scripts/train_models.py --cv-folds 10

# Train specific model
python scripts/train_models.py --algorithm random_forest

# Hyperparameter tuning
python scripts/train_models.py --tune-hyperparameters
```

### Evaluation
```bash
# Run comprehensive evaluation
python scripts/evaluate_models.py --detailed

# Compare with baseline
python scripts/evaluate_models.py --compare-baseline
```

## 🚀 Deployment

### REST API
```bash
# Start API server
python src/deployment/api.py --port 5000

# Test API
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Thank you for your application"}'
```

### Docker Deployment
```bash
# Build Docker image
docker build -t onlyjobs-classification .

# Run container
docker run -p 5000:5000 onlyjobs-classification
```

## 📈 Performance Comparison

### vs OpenAI GPT-3.5-turbo
| Metric | OnlyJobs ML | OpenAI LLM | Improvement |
|--------|-------------|------------|-------------|
| Accuracy | 90.15% | ~95% | -4.85% |
| Speed | 0.001s | 2.0s | **2000x faster** |
| Cost | $0.00 | $0.002/email | **100% savings** |
| Offline | ✅ Yes | ❌ No | **Available offline** |
| Privacy | ✅ Local | ❌ API | **Data stays local** |

### Annual Cost Savings
- **1K emails/month**: $24/year saved
- **10K emails/month**: $240/year saved  
- **100K emails/month**: $2,400/year saved

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_models.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

- [API Reference](docs/api_reference.md) - Complete API documentation
- [Data Collection Guide](docs/data_collection.md) - How to collect training data
- [Model Training Guide](docs/model_training.md) - Training and evaluation
- [Deployment Guide](docs/deployment.md) - Production deployment options

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- Gmail API integration via [Google API Client](https://github.com/googleapis/google-api-python-client)
- Inspired by the need for cost-effective email classification

## 📞 Support

- 📧 Email: support@onlyjobs-classification.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/onlyjobs-classification/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/onlyjobs-classification/discussions)

---

**Made with ❤️ for efficient email processing**