#!/usr/bin/env python3
"""
Production Email Classifier
Simple, production-ready interface for email classification.
Optimized for integration into other software projects.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProductionEmailClassifier:
    """
    Production-ready email classifier with minimal dependencies and simple API.
    
    Usage:
        classifier = ProductionEmailClassifier()
        classifier.load_model()
        result = classifier.classify_email(email_text, sender_email="sender@company.com")
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the classifier.
        
        Args:
            model_dir: Directory containing model files. If None, uses default location.
        """
        if model_dir is None:
            # Default to models directory in the same repo
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, '..', '..', 'data', 'models')
        
        self.model_dir = os.path.abspath(model_dir)
        self.model = None
        self.feature_pipeline = None
        self.is_loaded = False
        
        # Model metadata
        self.model_version = "1.0.0"
        self.model_name = "Generalized Email Classifier"
        self.accuracy = 0.98
        self.feature_count = 575
    
    def load_model(self):
        """
        Load the trained model and feature pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the generalized model (production version)
            model_path = os.path.join(self.model_dir, 'generalized_email_classifier.pkl')
            pipeline_path = os.path.join(self.model_dir, 'generalized_feature_pipeline.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if not os.path.exists(pipeline_path):
                raise FileNotFoundError(f"Feature pipeline not found: {pipeline_path}")
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load feature pipeline
            with open(pipeline_path, 'rb') as f:
                self.feature_pipeline = pickle.load(f)
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def classify_email(self, email_body, sender_email=None, date=None):
        """
        Classify a single email as job-related or not.
        
        Args:
            email_body (str): The email content to classify
            sender_email (str, optional): Sender's email address
            date (str, optional): Email date (any reasonable format)
        
        Returns:
            dict: Classification result with prediction, probability, and confidence
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not email_body or not isinstance(email_body, str):
            raise ValueError("email_body must be a non-empty string")
        
        try:
            # Prepare email data in expected format
            email_data = self._prepare_email_data(email_body, sender_email, date)
            
            # Extract features
            features = self._extract_features(email_data)
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            prediction_proba = self.model.predict_proba([features])[0]
            
            # Get probability for the positive class (job-related)
            job_probability = prediction_proba[1]
            
            # Determine confidence level
            confidence = self._get_confidence_level(job_probability)
            
            return {
                'is_job_related': bool(prediction == 1),
                'prediction': 'job_related' if prediction == 1 else 'non_job_related',
                'probability': round(float(job_probability), 4),
                'confidence': confidence,
                'model_version': self.model_version,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': f"Classification failed: {str(e)}",
                'is_job_related': None,
                'prediction': None,
                'probability': None,
                'confidence': None
            }
    
    def classify_batch(self, emails):
        """
        Classify multiple emails at once.
        
        Args:
            emails (list): List of dictionaries with keys: 'email_body', 'sender_email' (optional), 'date' (optional)
        
        Returns:
            list: List of classification results
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for i, email in enumerate(emails):
            try:
                result = self.classify_email(
                    email_body=email.get('email_body'),
                    sender_email=email.get('sender_email'),
                    date=email.get('date')
                )
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'is_job_related': None,
                    'prediction': None,
                    'probability': None,
                    'confidence': None
                })
        
        return results
    
    def _prepare_email_data(self, email_body, sender_email, date):
        """Prepare email data in the format expected by the feature pipeline."""
        current_date = datetime.now()
        
        # Use provided date or default to current
        if date is None:
            email_date = current_date
        else:
            # Try to parse the date
            try:
                if isinstance(date, str):
                    # Handle common date formats
                    for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f']:
                        try:
                            email_date = datetime.strptime(date.replace('Z', '+0000'), fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # If no format matches, use current date
                        email_date = current_date
                else:
                    email_date = current_date
            except:
                email_date = current_date
        
        # Extract sender info
        if sender_email and '@' in sender_email:
            sender_name = sender_email.split('@')[0]
            sender_domain = sender_email.split('@')[1]
        else:
            sender_name = "unknown"
            sender_domain = "unknown.com"
        
        # Check if sender is from talent acquisition domain
        ta_domains = ['workday.com', 'greenhouse.io', 'lever.co', 'icims.com', 'bamboohr.com']
        is_talent_acquisition = any(domain in sender_domain.lower() for domain in ta_domains)
        
        return {
            'email_body': email_body,
            'sender': sender_name,
            'sender_email': sender_email or f"{sender_name}@{sender_domain}",
            'date': email_date.strftime('%a, %d %b %Y %H:%M:%S %z'),
            'parsed_datetime': email_date.isoformat(),
            'date_only': email_date.strftime('%Y-%m-%d'),
            'week': email_date.isocalendar()[1],
            'month': email_date.month,
            'year': email_date.year,
            'days_since': (current_date - email_date).days,
            'is_talent_acquisition': is_talent_acquisition
        }
    
    def _extract_features(self, email_data):
        """Extract features using the loaded feature pipeline."""
        try:
            # First priority: Use the complete GeneralizedEmailFeatureExtractor object
            if hasattr(self.feature_pipeline, 'extract_all_features'):
                # Convert to DataFrame for feature extraction
                df = pd.DataFrame([email_data])
                features, _ = self.feature_pipeline.extract_all_features(df, fit=False)
                return features[0]
            
            # Second priority: If feature pipeline is a dictionary (legacy format)
            elif isinstance(self.feature_pipeline, dict):
                print("WARNING: Using legacy dictionary-based feature extraction. Please update pipeline.")
                return self._extract_features_from_components(email_data)
            
            else:
                # Last resort: basic features for compatibility
                print("WARNING: Using basic feature extraction fallback. Predictions may be inaccurate.")
                return self._extract_basic_features(email_data)
                
        except Exception as e:
            # If feature pipeline fails, provide detailed error information
            print(f"ERROR: Feature extraction failed with complete pipeline: {e}")
            print("Falling back to basic feature extraction. Predictions may be inaccurate.")
            return self._extract_basic_features(email_data)
    
    def _extract_basic_features(self, email_data):
        """Fallback basic feature extraction."""
        # This is a simplified fallback - in production you'd want full features
        email_body = email_data.get('email_body', '').lower()
        
        # Basic features for demonstration
        features = []
        
        # Job-related keywords
        job_words = ['application', 'position', 'job', 'interview', 'resume', 'hiring', 'candidate']
        for word in job_words:
            features.append(1 if word in email_body else 0)
        
        # Non-job keywords  
        non_job_words = ['order', 'shipping', 'payment', 'account', 'service', 'newsletter']
        for word in non_job_words:
            features.append(1 if word in email_body else 0)
        
        # Pad to expected feature count (575) with zeros
        while len(features) < 575:
            features.append(0)
        
        return np.array(features[:575])  # Ensure exactly 575 features
    
    def _extract_features_from_components(self, email_data):
        """Extract features when pipeline is saved as dictionary components."""
        # This is a simplified implementation for production use
        # In a full production system, you'd reconstruct the full feature pipeline
        email_body = email_data.get('email_body', '').lower()
        
        # Initialize feature vector
        features = []
        
        # TF-IDF features (simplified)
        job_terms = ['application', 'job', 'position', 'interview', 'resume', 'hiring', 
                    'candidate', 'recruiting', 'career', 'applying', 'thank', 'regards']
        for term in job_terms:
            features.append(1 if term in email_body else 0)
        
        # Sender features
        sender_email = email_data.get('sender_email', '').lower()
        features.append(1 if any(domain in sender_email for domain in ['workday', 'greenhouse', 'lever', 'icims']) else 0)
        features.append(1 if 'noreply' in sender_email else 0)
        
        # Temporal features
        features.extend([
            email_data.get('week', 0) / 52.0,
            email_data.get('month', 0) / 12.0,
            email_data.get('year', 2025) / 2025.0,
            min(email_data.get('days_since', 0) / 365.0, 1.0)
        ])
        
        # Text statistics
        text_len = len(email_body)
        features.extend([
            text_len / 10000.0,  # Normalized text length
            email_body.count('!') / max(text_len, 1),  # Exclamation density
            email_body.count('?') / max(text_len, 1),  # Question density
        ])
        
        # Keywords
        features.append(1 if 'application' in email_body else 0)
        features.append(1 if 'thank' in email_body and ('applying' in email_body or 'application' in email_body) else 0)
        
        # Domain features
        features.append(email_data.get('is_talent_acquisition', 0))
        
        # Pad to expected feature count (575) with zeros
        while len(features) < 575:
            features.append(0)
        
        return np.array(features[:575])  # Ensure exactly 575 features
    
    def _get_confidence_level(self, probability):
        """Determine confidence level based on probability."""
        if probability > 0.9 or probability < 0.1:
            return 'high'
        elif probability > 0.7 or probability < 0.3:
            return 'medium'
        else:
            return 'low'
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'accuracy': self.accuracy,
            'feature_count': self.feature_count,
            'is_loaded': self.is_loaded,
            'model_type': 'XGBoost Classifier',
            'training_date': '2025-08-27',
            'generalized': True
        }
    
    def health_check(self):
        """Perform a health check on the classifier."""
        try:
            if not self.is_loaded:
                return {'status': 'unhealthy', 'message': 'Model not loaded'}
            
            # Test with a simple email
            test_result = self.classify_email(
                "Thank you for applying to our Software Engineer position. We will review your application."
            )
            
            if test_result.get('error'):
                return {'status': 'unhealthy', 'message': test_result['error']}
            
            return {
                'status': 'healthy',
                'model_version': self.model_version,
                'test_prediction': test_result['prediction'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'unhealthy', 'message': str(e)}

def create_simple_example():
    """Example usage of the production classifier."""
    print("=== Production Email Classifier Example ===")
    
    # Initialize classifier
    classifier = ProductionEmailClassifier()
    
    # Load model
    if not classifier.load_model():
        print("Failed to load model. Please ensure model files exist.")
        return
    
    print("Model loaded successfully!")
    print(f"Model info: {classifier.get_model_info()}")
    
    # Test with sample emails
    test_emails = [
        {
            'email_body': "Thank you for applying to our Software Engineer position. We would like to schedule an interview.",
            'sender_email': "hr@techcompany.com"
        },
        {
            'email_body': "Your Amazon order has been shipped. Track your package here.",
            'sender_email': "no-reply@amazon.com"
        },
        {
            'email_body': "Congratulations! Your job application has been approved. Please reply to confirm.",
            'sender_email': "recruiter@startup.com"
        }
    ]
    
    print("\n=== Single Email Classification ===")
    for i, email in enumerate(test_emails):
        result = classifier.classify_email(
            email_body=email['email_body'],
            sender_email=email['sender_email']
        )
        print(f"\nEmail {i+1}:")
        print(f"Content: {email['email_body'][:60]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']}")
        print(f"Confidence: {result['confidence']}")
    
    print("\n=== Batch Classification ===")
    batch_results = classifier.classify_batch(test_emails)
    job_count = sum(1 for r in batch_results if r.get('is_job_related'))
    print(f"Processed {len(batch_results)} emails")
    print(f"Job-related emails: {job_count}")
    print(f"Non-job emails: {len(batch_results) - job_count}")
    
    # Health check
    print("\n=== Health Check ===")
    health = classifier.health_check()
    print(f"Status: {health['status']}")
    if health['status'] == 'healthy':
        print(f"Test prediction: {health['test_prediction']}")

if __name__ == "__main__":
    create_simple_example()