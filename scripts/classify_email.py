#!/usr/bin/env python3
"""
Production Email Classification Script
Simple command-line interface for the production email classifier.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.production_email_classifier import ProductionEmailClassifier

def classify_email_cli(email_text, sender_email=None):
    """Classify an email using the production classifier."""
    
    # Initialize classifier
    classifier = ProductionEmailClassifier()
    
    # Load model
    if not classifier.load_model():
        print("❌ Failed to load production model.")
        print("Please ensure model files exist in data/models/:")
        print("  - generalized_email_classifier.pkl")
        print("  - generalized_feature_pipeline.pkl")
        sys.exit(1)
    
    # Classify email
    result = classifier.classify_email(email_text, sender_email)
    
    if result.get('error'):
        print(f"❌ Classification failed: {result['error']}")
        sys.exit(1)
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Classify an email as job-related using the production model"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=True,
        help="Email text to classify"
    )
    parser.add_argument(
        "--sender", 
        type=str, 
        help="Sender email address (optional)"
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
        result = classify_email_cli(args.text, args.sender)
        
        if args.format == "json":
            import json
            print(json.dumps(result, indent=2))
        else:
            # Text format
            print("📧 Production Email Classification Result")
            print("="*50)
            print(f"Job-related: {'✅ YES' if result['is_job_related'] else '❌ NO'}")
            print(f"Probability: {result['probability']:.1%}")
            print(f"Confidence: {result['confidence'].upper()}")
            print(f"Model Version: {result['model_version']}")
            print("="*50)
            
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()