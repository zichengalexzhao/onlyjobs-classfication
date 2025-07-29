#!/usr/bin/env python3
"""
Optimized Job Email Classifier
Uses best performing features: Job Patterns + Domain Analysis
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path

# Add path for optimized feature pipeline
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / "feature_engineering"))

from optimized_feature_pipeline import OptimizedJobFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class OptimizedJobClassifier:
    def __init__(self):
        self.feature_extractor = OptimizedJobFeatureExtractor()
        self.model = None
        self.model_info = {
            'name': 'Optimized Random Forest',
            'features': 'Base + Job Patterns + Domain Analysis',
            'optimization': 'High Recall (3:1 class weight)',
            'version': '2.0'
        }
    
    def load_training_data(self):
        """Load training data"""
        print("📊 Loading training data...")
        
        train_path = "/Users/zichengzhao/Downloads/job-app-tracker/data/ml_training/train_data.json"
        val_path = "/Users/zichengzhao/Downloads/job-app-tracker/data/ml_training/val_data.json"
        test_path = "/Users/zichengzhao/Downloads/job-app-tracker/data/ml_training/test_data.json"
        
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        with open(val_path, 'r') as f:
            val_data = json.load(f)
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        # Combine train + val for training
        training_emails = train_data + val_data
        
        print(f"Training emails: {len(training_emails)}")
        print(f"Test emails: {len(test_data)}")
        
        return training_emails, test_data
    
    def train_optimized_model(self, training_emails):
        """Train the optimized model"""
        print("\\n🎯 Training Optimized Model")
        print("=" * 50)
        
        # Extract optimized features
        features = self.feature_extractor.extract_optimized_features(training_emails)
        
        # Extract labels
        labels = [1 if email['label'] == 'job_related' else 0 for email in training_emails]
        labels = np.array(labels)
        
        print(f"\\n📈 Training Configuration:")
        print(f"   Features: {features.shape[1]}")
        print(f"   Samples: {features.shape[0]}")
        print(f"   Positive examples: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        
        # Train Random Forest with high recall settings
        self.model = RandomForestClassifier(
            n_estimators=150,           # More trees for stability
            max_depth=20,               # Sufficient depth for complex patterns
            min_samples_split=2,        # Allow fine-grained splits
            min_samples_leaf=1,         # Allow detailed leaf nodes
            class_weight={0: 1.0, 1: 3.0},  # Heavy penalty for missing job emails
            bootstrap=True,             # Use bootstrap sampling
            random_state=42,
            n_jobs=-1
        )
        
        print("\\n🔄 Training Random Forest...")
        self.model.fit(features, labels)
        
        print("✅ Optimized model trained successfully!")
        
        return features, labels
    
    def evaluate_model(self, test_emails):
        """Evaluate model on test data"""
        print("\\n📊 Evaluating Optimized Model")
        print("=" * 50)
        
        # Extract features for test data
        print("Extracting test features...")
        test_features = self.feature_extractor.extract_optimized_features(test_emails)
        
        # Extract test labels
        test_labels = [1 if email['label'] == 'job_related' else 0 for email in test_emails]
        test_labels = np.array(test_labels)
        
        # Make predictions
        predictions = self.model.predict(test_features)
        probabilities = self.model.predict_proba(test_features)
        
        # Calculate detailed metrics
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"\\n📈 Model Performance:")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f} ⭐ (minimizes missed job emails)")
        print(f"   F1-Score:  {f1:.3f}")
        
        print(f"\\n🔍 Confusion Matrix:")
        print(f"   True Positives (TP):  {tp:3d} - Correctly identified job emails")
        print(f"   False Positives (FP): {fp:3d} - Non-job emails marked as job")
        print(f"   True Negatives (TN):  {tn:3d} - Correctly identified non-job emails")
        print(f"   False Negatives (FN): {fn:3d} - MISSED job emails ⚠️")
        
        # Classification report
        print(f"\\n📋 Detailed Classification Report:")
        print(classification_report(test_labels, predictions, 
                                   target_names=['Non-job', 'Job'], digits=3))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
        }
    
    def save_optimized_model(self, performance_metrics):
        """Save the optimized model"""
        print("\\n💾 Saving Optimized Model")
        print("=" * 30)
        
        os.makedirs("data/models", exist_ok=True)
        
        # Save model
        with open("data/models/optimized_job_classifier.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save feature extractors
        self.feature_extractor.save_optimized_extractors()
        
        # Save model information and performance
        model_info = {
            **self.model_info,
            'performance': performance_metrics,
            'features_total': self.model.n_features_in_,
            'training_samples': self.model.n_estimators,
            'class_weights': {0: 1.0, 1: 3.0}
        }
        
        with open("data/models/optimized_model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✅ Optimized model saved!")
        print("   Model: data/models/optimized_job_classifier.pkl")
        print("   Extractors: data/models/optimized_extractors.pkl")
        print("   Info: data/models/optimized_model_info.json")
    
    def classify_email(self, email_content):
        """Classify a single email"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_optimized_model() first.")
        
        # Create email object
        email_obj = {
            'full_content': email_content,
            'snippet': email_content[:200],
            'label': 'unknown'
        }
        
        # Extract features
        features = self.feature_extractor.extract_optimized_features([email_obj])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        return {
            'is_job_related': bool(prediction),
            'confidence': float(max(probabilities)),
            'job_probability': float(probabilities[1]),
            'non_job_probability': float(probabilities[0])
        }
    
    def train_and_evaluate(self):
        """Complete training and evaluation pipeline"""
        print("🚀 Optimized Job Email Classifier Training")
        print("=" * 60)
        
        # Load data
        training_emails, test_emails = self.load_training_data()
        
        # Train model
        features, labels = self.train_optimized_model(training_emails)
        
        # Evaluate model
        performance = self.evaluate_model(test_emails)
        
        # Save model
        self.save_optimized_model(performance)
        
        print(f"\\n🎉 Training Complete!")
        print(f"   Best feature: Job Pattern Detection")
        print(f"   Recall: {performance['recall']:.1%} (minimizes missed job emails)")
        print(f"   Model ready for production use!")
        
        return True

def main():
    """Train the optimized classifier"""
    classifier = OptimizedJobClassifier()
    success = classifier.train_and_evaluate()
    
    if success:
        print("\\n✅ Optimized classifier ready!")
        
        # Test with a sample email
        sample_email = """Subject: Thank you for applying to TechCorp
        
        Dear Alex,
        
        Thank you for applying to the Data Scientist position at TechCorp.
        We have received your application and our hiring team will review it.
        
        If your qualifications match our requirements, we will contact you
        for the next steps in our interview process.
        
        Best regards,
        TechCorp Recruiting Team"""
        
        result = classifier.classify_email(sample_email)
        print(f"\\n🧪 Sample Classification:")
        print(f"   Job-related: {result['is_job_related']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        
    else:
        print("\\n❌ Training failed")

if __name__ == "__main__":
    main()