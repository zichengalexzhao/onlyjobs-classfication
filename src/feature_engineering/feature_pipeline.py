#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Email Classification
Extracts meaningful features from email content for ML model training.
"""

import json
import os
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime

class EmailFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scaler = None
        self.job_keywords = [
            'application', 'position', 'role', 'interview', 'candidate', 'hiring', 
            'job', 'career', 'resume', 'cv', 'employment', 'opportunity', 'company',
            'thank', 'applied', 'consideration', 'interested', 'qualified', 'skills',
            'experience', 'background', 'team', 'join', 'offer', 'salary', 'benefits'
        ]
        
        self.rejection_keywords = [
            'unfortunately', 'declined', 'rejected', 'not selected', 'other candidates',
            'different direction', 'not moving forward', 'regret to inform'
        ]
        
        self.interview_keywords = [
            'interview', 'meeting', 'schedule', 'phone call', 'video call', 'zoom',
            'calendar', 'availability', 'next step', 'discuss', 'conversation'
        ]
    
    def extract_text_features(self, emails):
        """Extract text-based features from emails."""
        print("📝 Extracting text features...")
        
        features = []
        
        for email in emails:
            content = email.get('full_content', '') or email.get('content', '')
            snippet = email.get('snippet', '')
            
            # Combine snippet and content for full text analysis
            full_text = f"{snippet} {content}".lower()
            
            # Basic text statistics
            text_features = {
                'content_length': len(content),
                'snippet_length': len(snippet),
                'word_count': len(full_text.split()),
                'sentence_count': len(re.findall(r'[.!?]+', full_text)),
                'exclamation_count': full_text.count('!'),
                'question_count': full_text.count('?'),
                'capital_ratio': sum(1 for c in content if c.isupper()) / max(len(content), 1),
                'digit_count': sum(1 for c in full_text if c.isdigit()),
                'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', full_text)),
                'email_count': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_text))
            }
            
            features.append(text_features)
        
        return features
    
    def extract_keyword_features(self, emails):
        """Extract keyword-based features."""
        print("🔑 Extracting keyword features...")
        
        features = []
        
        for email in emails:
            content = email.get('full_content', '') or email.get('content', '')
            snippet = email.get('snippet', '')
            full_text = f"{snippet} {content}".lower()
            
            keyword_features = {
                'job_keyword_count': sum(1 for keyword in self.job_keywords if keyword in full_text),
                'rejection_keyword_count': sum(1 for keyword in self.rejection_keywords if keyword in full_text),
                'interview_keyword_count': sum(1 for keyword in self.interview_keywords if keyword in full_text),
                'has_thank_you': int('thank' in full_text),
                'has_application': int('application' in full_text),
                'has_position': int('position' in full_text or 'role' in full_text),
                'has_company_name': int(any(word in full_text for word in ['company', 'corporation', 'inc', 'llc'])),
                'has_greeting': int(any(greeting in full_text for greeting in ['dear', 'hello', 'hi', 'greetings'])),
                'has_signature': int(any(sig in full_text for sig in ['best regards', 'sincerely', 'best', 'regards']))
            }
            
            features.append(keyword_features)
        
        return features
    
    def extract_structural_features(self, emails):
        """Extract email structure-based features."""
        print("🏗️ Extracting structural features...")
        
        features = []
        
        for email in emails:
            content = email.get('full_content', '') or email.get('content', '')
            
            structural_features = {
                'has_html': int('<' in content and '>' in content),
                'html_tag_count': len(re.findall(r'<[^>]+>', content)),
                'line_count': len(content.split('\n')),
                'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
                'has_styling': int('style=' in content.lower()),
                'has_links': int('http' in content.lower()),
                'has_images': int('img' in content.lower() or 'image' in content.lower()),
                'has_formatting': int(any(tag in content.lower() for tag in ['<b>', '<i>', '<u>', '<strong>', '<em>'])),
                'whitespace_ratio': (content.count(' ') + content.count('\n') + content.count('\t')) / max(len(content), 1)
            }
            
            features.append(structural_features)
        
        return features
    
    def extract_tfidf_features(self, emails, max_features=1000):
        """Extract TF-IDF features from email content."""
        print(f"📊 Extracting TF-IDF features (max_features={max_features})...")
        
        # Prepare texts
        texts = []
        for email in emails:
            content = email.get('full_content', '') or email.get('content', '')
            snippet = email.get('snippet', '')
            # Clean and combine text
            full_text = f"{snippet} {content}"
            # Remove HTML tags and normalize
            clean_text = re.sub(r'<[^>]+>', ' ', full_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            texts.append(clean_text)
        
        # Initialize or use existing TF-IDF vectorizer
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.95,  # Ignore terms that appear in more than 95% of documents
                lowercase=True,
                strip_accents='ascii'
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix.toarray()
    
    def combine_features(self, text_features, keyword_features, structural_features, tfidf_features):
        """Combine all feature types into a single feature matrix."""
        print("🔗 Combining all features...")
        
        # Convert to numpy arrays
        text_array = np.array([[f[key] for key in sorted(f.keys())] for f in text_features])
        keyword_array = np.array([[f[key] for key in sorted(f.keys())] for f in keyword_features])
        structural_array = np.array([[f[key] for key in sorted(f.keys())] for f in structural_features])
        
        # Combine all features
        combined_features = np.hstack([
            text_array,
            keyword_array,
            structural_array,
            tfidf_features
        ])
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(combined_features)
        else:
            scaled_features = self.scaler.transform(combined_features)
        
        print(f"✅ Combined feature matrix shape: {scaled_features.shape}")
        
        return scaled_features
    
    def extract_all_features(self, emails, max_tfidf_features=1000):
        """Extract all features from emails."""
        print(f"\n🔧 Feature Engineering Pipeline")
        print(f"Processing {len(emails)} emails...")
        print("="*50)
        
        # Extract different types of features
        text_features = self.extract_text_features(emails)
        keyword_features = self.extract_keyword_features(emails)
        structural_features = self.extract_structural_features(emails)
        tfidf_features = self.extract_tfidf_features(emails, max_tfidf_features)
        
        # Combine all features
        combined_features = self.combine_features(
            text_features, keyword_features, structural_features, tfidf_features
        )
        
        # Feature names for interpretability
        feature_names = self._get_feature_names(text_features, keyword_features, structural_features)
        
        print(f"✅ Feature extraction complete!")
        print(f"   Total features: {combined_features.shape[1]}")
        print(f"   Feature types: text({len(text_features[0])}), keywords({len(keyword_features[0])}), structural({len(structural_features[0])}), tfidf({tfidf_features.shape[1]})")
        
        return combined_features, feature_names
    
    def _get_feature_names(self, text_features, keyword_features, structural_features):
        """Get feature names for interpretability."""
        names = []
        
        # Text feature names
        names.extend([f"text_{key}" for key in sorted(text_features[0].keys())])
        
        # Keyword feature names
        names.extend([f"keyword_{key}" for key in sorted(keyword_features[0].keys())])
        
        # Structural feature names
        names.extend([f"structural_{key}" for key in sorted(structural_features[0].keys())])
        
        # TF-IDF feature names
        if self.tfidf_vectorizer:
            names.extend([f"tfidf_{term}" for term in self.tfidf_vectorizer.get_feature_names_out()])
        
        return names
    
    def save_feature_extractors(self, filepath="data/models/feature_extractors.pkl"):
        """Save trained feature extractors."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        extractors = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'job_keywords': self.job_keywords,
            'rejection_keywords': self.rejection_keywords,
            'interview_keywords': self.interview_keywords
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(extractors, f)
        
        print(f"💾 Feature extractors saved to: {filepath}")
    
    def load_feature_extractors(self, filepath="data/models/feature_extractors.pkl"):
        """Load trained feature extractors."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                extractors = pickle.load(f)
            
            self.tfidf_vectorizer = extractors['tfidf_vectorizer']
            self.scaler = extractors['scaler']
            self.job_keywords = extractors['job_keywords']
            self.rejection_keywords = extractors['rejection_keywords']
            self.interview_keywords = extractors['interview_keywords']
            
            print(f"📂 Feature extractors loaded from: {filepath}")
            return True
        return False

def process_dataset_features():
    """Process features for the entire dataset."""
    print("🚀 Processing Dataset Features")
    print("="*60)
    
    # Load real email dataset
    data_path = "../data/real_email_training"
    
    datasets = {}
    for split in ['train_data', 'val_data', 'test_data']:
        filepath = os.path.join(data_path, f"{split}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                datasets[split] = json.load(f)
    
    if not datasets:
        print("❌ No dataset found. Please run data collection first.")
        return
    
    # Initialize feature extractor
    feature_extractor = EmailFeatureExtractor()
    
    # Process each dataset split
    processed_data = {}
    
    for split_name, emails in datasets.items():
        print(f"\n📊 Processing {split_name}...")
        
        # Extract features
        features, feature_names = feature_extractor.extract_all_features(emails)
        
        # Extract labels
        labels = np.array([1 if email.get('label') == 'job_related' else 0 for email in emails])
        
        processed_data[split_name] = {
            'features': features,
            'labels': labels,
            'feature_names': feature_names,
            'original_data': emails
        }
        
        print(f"✅ {split_name}: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Save processed features
    os.makedirs("data/models", exist_ok=True)
    
    with open("data/models/processed_features.pkl", 'wb') as f:
        pickle.dump(processed_data, f)
    
    # Save feature extractors
    feature_extractor.save_feature_extractors()
    
    # Save feature analysis
    feature_analysis = {
        'total_features': len(feature_names),
        'feature_names': feature_names,
        'dataset_shapes': {split: data['features'].shape for split, data in processed_data.items()},
        'processing_date': datetime.now().isoformat()
    }
    
    with open("data/models/feature_analysis.json", 'w') as f:
        json.dump(feature_analysis, f, indent=2)
    
    print(f"\n🎉 Feature Processing Complete!")
    print(f"📁 Saved to: data/models/")
    print(f"   - processed_features.pkl (features & labels)")
    print(f"   - feature_extractors.pkl (trained extractors)")
    print(f"   - feature_analysis.json (feature metadata)")
    
    # Feature summary
    print(f"\n📊 Feature Summary:")
    for split_name, data in processed_data.items():
        pos_samples = np.sum(data['labels'])
        neg_samples = len(data['labels']) - pos_samples
        print(f"   {split_name}: {data['features'].shape[0]} samples ({pos_samples} job, {neg_samples} non-job)")
    
    print(f"   Total features: {len(feature_names)}")
    print(f"\n🚀 Ready for ML model training!")

if __name__ == "__main__":
    process_dataset_features()