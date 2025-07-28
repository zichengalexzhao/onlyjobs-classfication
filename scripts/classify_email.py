#!/usr/bin/env python3
"""
Email Classification Script
Classifies a single email as job-related or not.
"""

import argparse
import sys
import pickle
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feature_engineering.feature_pipeline import EmailFeatureExtractor

def load_model(model_path="data/models/best_model.pkl"):
    """Load the trained model."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Run 'python scripts/train_models.py' first to train a model.")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def classify_email(email_text, model_path="data/models/best_model.pkl"):
    """Classify a single email."""
    
    # Load model
    model = load_model(model_path)
    
    # Initialize feature extractor
    feature_extractor = EmailFeatureExtractor()
    
    # Load feature extractors
    extractor_path = "data/models/feature_extractors.pkl"
    if not feature_extractor.load_feature_extractors(extractor_path):
        print(f"❌ Feature extractors not found at {extractor_path}")
        sys.exit(1)
    
    # Create email object
    email_obj = {
        'full_content': email_text,
        'snippet': email_text[:200],  # First 200 chars as snippet
        'label': 'unknown'  # Placeholder
    }
    
    # Extract features
    features, _ = feature_extractor.extract_all_features([email_obj])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
    
    return {
        'is_job_related': bool(prediction),
        'confidence': float(max(probability)),
        'probabilities': {
            'non_job_related': float(probability[0]),
            'job_related': float(probability[1])
        }
    }

def main():
    parser = argparse.ArgumentParser(
        description="Classify an email as job-related or not"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=True,
        help="Email text to classify"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="data/models/best_model.pkl",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["json", "text"],
        default="text",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    try:
        # Classify email
        result = classify_email(args.text, args.model)
        
        if args.format == "json":
            import json
            print(json.dumps(result, indent=2))
        else:
            # Text format
            print("📧 Email Classification Result")
            print("="*40)
            print(f"Job-related: {'✅ YES' if result['is_job_related'] else '❌ NO'}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"\nProbabilities:")
            print(f"  Non-job: {result['probabilities']['non_job_related']:.2%}")
            print(f"  Job-related: {result['probabilities']['job_related']:.2%}")
            
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()