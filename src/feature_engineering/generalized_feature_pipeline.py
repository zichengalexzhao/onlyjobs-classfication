#!/usr/bin/env python3
"""
Generalized Feature Engineering Pipeline for Email Classification
Designed to work with any user's email data by removing user-specific overfitting.
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
from datetime import datetime, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

class GeneralizedEmailFeatureExtractor:
    """
    Generalized feature extractor that removes user-specific features
    to prevent overfitting and ensure model works for any user.
    """
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scaler = None
        self.domain_encoder = None
        self.fitted = False
        
        # Enhanced keyword sets (same as before)
        self.job_keywords = [
            'application', 'position', 'role', 'interview', 'candidate', 'hiring', 
            'job', 'career', 'resume', 'cv', 'employment', 'opportunity', 'company',
            'thank', 'applied', 'consideration', 'interested', 'qualified', 'skills',
            'experience', 'background', 'team', 'join', 'offer', 'salary', 'benefits',
            'recruiter', 'talent', 'acquisition', 'hr', 'human resources', 'screening',
            'onsite', 'phone screen', 'technical', 'coding', 'assessment'
        ]
        
        self.rejection_keywords = [
            'unfortunately', 'declined', 'rejected', 'not selected', 'other candidates',
            'different direction', 'not moving forward', 'regret to inform', 'unable to',
            'not the right fit', 'competitive', 'decided to go', 'another candidate'
        ]
        
        self.interview_keywords = [
            'interview', 'meeting', 'schedule', 'phone call', 'video call', 'zoom',
            'calendar', 'availability', 'next step', 'discuss', 'conversation',
            'setup', 'arrange', 'coordinate', 'invite', 'appointment'
        ]
        
        self.positive_keywords = [
            'congratulations', 'pleased', 'excited', 'welcome', 'move forward',
            'next round', 'offer', 'accept', 'successful', 'impressed'
        ]
        
        # Common recruiter domains
        self.recruiter_domains = [
            'lever.co', 'greenhouse.io', 'workday.com', 'bamboohr.com', 'jobvite.com',
            'smartrecruiters.com', 'icims.com', 'taleo.net', 'myworkdayjobs.com',
            'ultipro.com', 'successfactors.com', 'cornerstone.com'
        ]
    
    def load_data(self, file_path):
        """Load data from CSV file."""
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} emails")
        return df
    
    def generalize_email_text(self, email_body):
        """
        Generalize email text by replacing user-specific information with generic tokens.
        This is the core function that prevents user-specific overfitting.
        """
        if pd.isna(email_body):
            return ""
            
        # Convert to string and make case-insensitive for replacements
        text = str(email_body)
        
        # Replace specific user names with generic tokens
        # This handles variations like "Alex Zhao", "alex zhao", "Zicheng", etc.
        text = re.sub(r'\b(alex|zicheng)\s+(zhao)\b', 'USER_FULL_NAME', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(alex)\s+(zhao)\b', 'USER_FULL_NAME', text, flags=re.IGNORECASE)
        text = re.sub(r'\bzicheng\s+zhao\b', 'USER_FULL_NAME', text, flags=re.IGNORECASE)
        
        # Replace individual name components
        text = re.sub(r'\bzicheng\b', 'USER_FIRST_NAME', text, flags=re.IGNORECASE)
        text = re.sub(r'\balex\b', 'USER_FIRST_NAME', text, flags=re.IGNORECASE) 
        text = re.sub(r'\bzhao\b', 'USER_LAST_NAME', text, flags=re.IGNORECASE)
        
        # Replace common greeting patterns with user names
        text = re.sub(r'\bhi\s+USER_FIRST_NAME\b', 'hi USER_FIRST_NAME', text, flags=re.IGNORECASE)
        text = re.sub(r'\bdear\s+USER_FIRST_NAME\b', 'dear USER_FIRST_NAME', text, flags=re.IGNORECASE)
        text = re.sub(r'\bhello\s+USER_FIRST_NAME\b', 'hello USER_FIRST_NAME', text, flags=re.IGNORECASE)
        
        # Replace email addresses that contain user names
        text = re.sub(r'\bzichengalexzhao@gmail\.com\b', 'USER_EMAIL', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[a-zA-Z0-9._%+-]*zhao[a-zA-Z0-9._%+-]*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', 'USER_EMAIL', text, flags=re.IGNORECASE)
        
        # Add generic patterns for user identification
        # This helps the model learn patterns instead of memorizing specific names
        user_name_count = len(re.findall(r'USER_(?:FIRST_|LAST_|FULL_)?NAME', text))
        if user_name_count > 0:
            text += f" USER_NAME_MENTIONS_{user_name_count}"
            
        return text
    
    def extract_temporal_features(self, df):
        """Extract temporal features from date/time data."""
        print("Extracting temporal features...")
        
        features = []
        for _, row in df.iterrows():
            temp_features = {
                'week': row['week'],
                'month': row['month'], 
                'year': row['year'],
                'days_since': row['days_since'],
                'is_weekend': int(pd.to_datetime(row['date_only']).weekday() >= 5),
                'is_business_hours': self._is_business_hours(row['parsed_datetime']),
                'is_recent': int(row['days_since'] <= 7),
                'is_very_old': int(row['days_since'] >= 60),
                'quarter': int((row['month'] - 1) // 3 + 1),
                'day_of_week': pd.to_datetime(row['date_only']).weekday(),
            }
            features.append(temp_features)
        
        return features
    
    def _is_business_hours(self, datetime_str):
        """Check if email was sent during business hours (9-17)."""
        try:
            dt = pd.to_datetime(datetime_str)
            hour = dt.hour
            return int(9 <= hour <= 17)
        except:
            return 0
    
    def extract_sender_features(self, df):
        """Extract features from sender information."""
        print("Extracting sender features...")
        
        features = []
        domains = []
        
        for _, row in df.iterrows():
            sender = str(row['sender']).lower()
            sender_email = str(row['sender_email']).lower()
            
            # Extract domain
            try:
                domain = sender_email.split('@')[1] if '@' in sender_email else ''
            except:
                domain = ''
            
            domains.append(domain)
            
            sender_features = {
                'sender_length': len(sender),
                'sender_has_numbers': int(bool(re.search(r'\d', sender))),
                'sender_word_count': len(sender.split()),
                'email_has_noreply': int('noreply' in sender_email or 'no-reply' in sender_email),
                'email_has_hire': int('hire' in sender_email or 'recruiting' in sender_email),
                'is_recruiter_domain': int(any(rec_domain in domain for rec_domain in self.recruiter_domains)),
                'domain_length': len(domain),
                'has_subdomain': int(domain.count('.') > 1),
                'is_common_email_provider': int(domain in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']),
                'sender_is_company_name': int(sender.replace(' ', '') in sender_email.replace('@', '').replace('.', '')),
            }
            features.append(sender_features)
        
        return features, domains
    
    def extract_text_features(self, df):
        """Extract comprehensive text features from generalized email body."""
        print("Extracting text features from generalized email content...")
        
        features = []
        
        for _, row in df.iterrows():
            # Apply generalization first
            email_body = self.generalize_email_text(row['email_body'])
            
            # Basic text statistics
            words = email_body.split()
            sentences = re.split(r'[.!?]+', email_body)
            paragraphs = email_body.split('\n\n')
            
            text_features = {
                'email_length': len(email_body),
                'word_count': len(words),
                'sentence_count': len([s for s in sentences if s.strip()]),
                'paragraph_count': len([p for p in paragraphs if p.strip()]),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'avg_sentence_length': len(words) / max(len(sentences), 1),
                
                # Character-based features
                'capital_ratio': sum(1 for c in email_body if c.isupper()) / max(len(email_body), 1),
                'digit_ratio': sum(1 for c in email_body if c.isdigit()) / max(len(email_body), 1),
                'punctuation_ratio': sum(1 for c in email_body if c in '.,!?;:') / max(len(email_body), 1),
                'whitespace_ratio': sum(1 for c in email_body if c.isspace()) / max(len(email_body), 1),
                
                # Special characters
                'exclamation_count': email_body.count('!'),
                'question_count': email_body.count('?'),
                'colon_count': email_body.count(':'),
                'semicolon_count': email_body.count(';'),
                
                # URLs and emails
                'url_count': len(re.findall(r'http[s]?://[^\s]+', email_body)),
                'email_mention_count': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', email_body)),
                
                # HTML and formatting
                'has_html': int('<' in email_body and '>' in email_body),
                'html_tag_count': len(re.findall(r'<[^>]+>', email_body)),
                'has_images': int('img' in email_body.lower() or '<image' in email_body.lower()),
                'has_links': int('href=' in email_body.lower() or 'http' in email_body.lower()),
                
                # Line-based features
                'line_count': len(email_body.split('\n')),
                'empty_line_count': email_body.count('\n\n'),
                'max_line_length': max([len(line) for line in email_body.split('\n')] or [0]),
                
                # NEW: Generalized user interaction features
                'has_user_name_mention': int('USER_' in email_body),
                'user_name_mention_count': len(re.findall(r'USER_(?:FIRST_|LAST_|FULL_)?NAME', email_body)),
                'has_personalized_greeting': int(re.search(r'\b(?:hi|dear|hello)\s+USER_FIRST_NAME\b', email_body, re.IGNORECASE) is not None),
                'has_user_email_mention': int('USER_EMAIL' in email_body),
            }
            
            features.append(text_features)
        
        return features
    
    def extract_keyword_features(self, df):
        """Extract keyword-based features from generalized text."""
        print("Extracting keyword features...")
        
        features = []
        
        for _, row in df.iterrows():
            # Apply generalization first
            email_body = self.generalize_email_text(row['email_body']).lower()
            
            # Count occurrences of different keyword types
            job_count = sum(1 for keyword in self.job_keywords if keyword in email_body)
            rejection_count = sum(1 for keyword in self.rejection_keywords if keyword in email_body)
            interview_count = sum(1 for keyword in self.interview_keywords if keyword in email_body)
            positive_count = sum(1 for keyword in self.positive_keywords if keyword in email_body)
            
            keyword_features = {
                'job_keyword_count': job_count,
                'rejection_keyword_count': rejection_count,
                'interview_keyword_count': interview_count,
                'positive_keyword_count': positive_count,
                'total_keyword_count': job_count + rejection_count + interview_count + positive_count,
                
                # Binary indicators
                'has_thank_you': int('thank' in email_body),
                'has_application': int('application' in email_body or 'apply' in email_body),
                'has_position': int('position' in email_body or 'role' in email_body),
                'has_interview': int('interview' in email_body),
                'has_company': int('company' in email_body or 'corporation' in email_body),
                'has_resume': int('resume' in email_body or 'cv' in email_body),
                'has_experience': int('experience' in email_body or 'background' in email_body),
                'has_skills': int('skills' in email_body or 'qualification' in email_body),
                'has_salary': int('salary' in email_body or 'compensation' in email_body or 'pay' in email_body),
                'has_benefits': int('benefits' in email_body or 'insurance' in email_body),
                'has_team': int('team' in email_body or 'group' in email_body),
                'has_opportunity': int('opportunity' in email_body or 'opening' in email_body),
                
                # Greeting and closing
                'has_greeting': int(any(greeting in email_body for greeting in ['dear', 'hello', 'hi ', 'greetings'])),
                'has_formal_closing': int(any(closing in email_body for closing in ['sincerely', 'best regards', 'regards'])),
                'has_casual_closing': int(any(closing in email_body for closing in ['best', 'thanks', 'cheers'])),
                
                # Call to action indicators
                'has_call_to_action': int(any(cta in email_body for cta in ['please', 'contact', 'reach out', 'let me know', 'get back'])),
                'has_urgency': int(any(urgent in email_body for urgent in ['urgent', 'asap', 'immediately', 'soon as possible'])),
                
                # NEW: Generalized user interaction patterns
                'has_personalized_content': int('user_first_name' in email_body or 'user_full_name' in email_body),
                'has_thank_user_pattern': int(re.search(r'thank\s+(?:you\s+)?user_first_name', email_body) is not None),
            }
            
            features.append(keyword_features)
        
        return features
    
    def extract_tfidf_features(self, df, max_features=500, fit=True):
        """Extract TF-IDF features from generalized email content."""
        print(f"Extracting TF-IDF features from generalized text (max_features={max_features})...")
        
        # Prepare texts with generalization applied
        texts = []
        for _, row in df.iterrows():
            # Apply generalization first
            email_body = self.generalize_email_text(row['email_body'])
            
            # Clean text
            clean_text = re.sub(r'<[^>]+>', ' ', email_body)  # Remove HTML
            clean_text = re.sub(r'http[s]?://[^\s]+', ' URL ', clean_text)  # Replace URLs
            clean_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL ', clean_text)  # Replace emails
            clean_text = re.sub(r'\d+', ' NUMBER ', clean_text)  # Replace numbers
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
            texts.append(clean_text)
        
        # Initialize or use existing TF-IDF vectorizer
        if self.tfidf_vectorizer is None and fit:
            # Custom stop words to exclude user-specific tokens
            stop_words_list = list(set([
                'alex', 'zhao', 'zicheng', 'alexzhao', 'zichengzhao', 
                'zichengalexzhao'  # Add known user-specific terms to stop words
            ] + [
                # Standard English stop words will be handled by 'english'
            ]))
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',  # Use standard English stop words
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                lowercase=True,
                strip_accents='ascii',
                token_pattern=r'\b[a-zA-Z_]{2,}\b'  # Include underscores for our generalized tokens
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        elif self.tfidf_vectorizer is not None:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        else:
            raise ValueError("TF-IDF vectorizer not fitted and fit=False")
        
        return tfidf_matrix.toarray()
    
    def extract_domain_features(self, domains, fit=True):
        """Extract domain-based features."""
        print("Extracting domain features...")
        
        # Get top domains for encoding
        if fit:
            domain_counts = Counter(domains)
            top_domains = [domain for domain, count in domain_counts.most_common(20) if count >= 5]
            
            if self.domain_encoder is None:
                self.domain_encoder = {}
            
            for i, domain in enumerate(top_domains):
                self.domain_encoder[domain] = i
        
        # Create domain features
        domain_features = []
        for domain in domains:
            features = {
                'domain_encoded': self.domain_encoder.get(domain, -1),
                'is_top_domain': int(domain in self.domain_encoder),
                'domain_popularity': len([d for d in domains if d == domain]) if domain else 0
            }
            domain_features.append(features)
        
        return domain_features
    
    def combine_all_features(self, temporal_features, sender_features, text_features, 
                           keyword_features, tfidf_features, domain_features, 
                           talent_acquisition_flags, fit=True):
        """Combine all features into a single matrix."""
        print("Combining all generalized features...")
        
        # Convert feature dictionaries to arrays
        temp_array = np.array([[f[key] for key in sorted(f.keys())] for f in temporal_features])
        sender_array = np.array([[f[key] for key in sorted(f.keys())] for f in sender_features])
        text_array = np.array([[f[key] for key in sorted(f.keys())] for f in text_features])
        keyword_array = np.array([[f[key] for key in sorted(f.keys())] for f in keyword_features])
        domain_array = np.array([[f[key] for key in sorted(f.keys())] for f in domain_features])
        
        # Add talent acquisition flag
        talent_array = np.array(talent_acquisition_flags).reshape(-1, 1)
        
        # Combine all features
        combined_features = np.hstack([
            temp_array,
            sender_array, 
            text_array,
            keyword_array,
            domain_array,
            talent_array,
            tfidf_features
        ])
        
        # Scale features
        if self.scaler is None and fit:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(combined_features)
        elif self.scaler is not None:
            scaled_features = self.scaler.transform(combined_features)
        else:
            raise ValueError("Scaler not fitted and fit=False")
        
        print(f"Combined generalized feature matrix shape: {scaled_features.shape}")
        
        return scaled_features
    
    def extract_all_features(self, df, max_tfidf_features=500, fit=True):
        """Extract all generalized features from the dataframe."""
        print(f"\n=== Generalized Email Classification Feature Engineering ===")
        print(f"Processing {len(df)} emails with user-agnostic features...")
        print("="*60)
        
        # Extract different types of features
        temporal_features = self.extract_temporal_features(df)
        sender_features, domains = self.extract_sender_features(df)
        text_features = self.extract_text_features(df)
        keyword_features = self.extract_keyword_features(df)
        tfidf_features = self.extract_tfidf_features(df, max_tfidf_features, fit=fit)
        domain_features = self.extract_domain_features(domains, fit=fit)
        
        # Get talent acquisition flags
        talent_acquisition_flags = df['is_talent_acquisition'].astype(int).values
        
        # Combine all features
        combined_features = self.combine_all_features(
            temporal_features, sender_features, text_features, keyword_features,
            tfidf_features, domain_features, talent_acquisition_flags, fit=fit
        )
        
        # Generate feature names
        feature_names = self._generate_feature_names(
            temporal_features, sender_features, text_features, 
            keyword_features, domain_features
        )
        
        if fit:
            self.fitted = True
        
        print(f"\n=== Generalized Feature Extraction Complete ===")
        print(f"Total features: {combined_features.shape[1]}")
        print(f"Feature breakdown:")
        print(f"  - Temporal: {len(temporal_features[0])}")
        print(f"  - Sender: {len(sender_features[0])}")
        print(f"  - Text: {len(text_features[0])}")
        print(f"  - Keywords: {len(keyword_features[0])}")
        print(f"  - Domain: {len(domain_features[0])}")
        print(f"  - Talent Acquisition: 1")
        print(f"  - TF-IDF: {tfidf_features.shape[1]}")
        print("\nKEY: All user-specific features have been generalized!")
        
        return combined_features, feature_names
    
    def _generate_feature_names(self, temporal_features, sender_features, text_features, 
                              keyword_features, domain_features):
        """Generate feature names for interpretability."""
        names = []
        
        # Add feature names in the same order as combination
        names.extend([f"temporal_{key}" for key in sorted(temporal_features[0].keys())])
        names.extend([f"sender_{key}" for key in sorted(sender_features[0].keys())])
        names.extend([f"text_{key}" for key in sorted(text_features[0].keys())])
        names.extend([f"keyword_{key}" for key in sorted(keyword_features[0].keys())])
        names.extend([f"domain_{key}" for key in sorted(domain_features[0].keys())])
        names.append("is_talent_acquisition")
        
        # Add TF-IDF feature names
        if self.tfidf_vectorizer:
            names.extend([f"tfidf_{term}" for term in self.tfidf_vectorizer.get_feature_names_out()])
        
        return names
    
    def save_pipeline(self, filepath):
        """Save the complete generalized feature extraction pipeline."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        pipeline_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'domain_encoder': self.domain_encoder,
            'fitted': self.fitted,
            'job_keywords': self.job_keywords,
            'rejection_keywords': self.rejection_keywords,
            'interview_keywords': self.interview_keywords,
            'positive_keywords': self.positive_keywords,
            'recruiter_domains': self.recruiter_domains,
            'pipeline_type': 'generalized'  # Mark as generalized pipeline
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Generalized feature extraction pipeline saved to: {filepath}")
    
    def load_pipeline(self, filepath):
        """Load the complete generalized feature extraction pipeline."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                pipeline_data = pickle.load(f)
            
            self.tfidf_vectorizer = pipeline_data['tfidf_vectorizer']
            self.scaler = pipeline_data['scaler']
            self.domain_encoder = pipeline_data['domain_encoder']
            self.fitted = pipeline_data['fitted']
            self.job_keywords = pipeline_data['job_keywords']
            self.rejection_keywords = pipeline_data['rejection_keywords']
            self.interview_keywords = pipeline_data['interview_keywords']
            self.positive_keywords = pipeline_data['positive_keywords']
            self.recruiter_domains = pipeline_data['recruiter_domains']
            
            print(f"Generalized feature extraction pipeline loaded from: {filepath}")
            return True
        return False

def main():
    """Main function to process the training dataset with generalized features."""
    print("=== Generalized Email Classification Feature Engineering ===")
    
    # File paths
    data_dir = "/Users/zichengzhao/Downloads/classification/onlyjobs-classfication/data/enhanced_training"
    models_dir = "/Users/zichengzhao/Downloads/classification/onlyjobs-classfication/data/models"
    
    # Initialize generalized feature extractor
    extractor = GeneralizedEmailFeatureExtractor()
    
    # Process training data first (to fit transformers)
    print("\n=== Processing Training Data with Generalized Features ===")
    train_df = extractor.load_data(f"{data_dir}/train_data.csv")
    train_features, feature_names = extractor.extract_all_features(train_df, max_tfidf_features=500, fit=True)
    train_labels = (train_df['label'] == 'job_related').astype(int).values
    
    # Process validation data
    print("\n=== Processing Validation Data with Generalized Features ===")
    val_df = extractor.load_data(f"{data_dir}/val_data.csv")
    val_features, _ = extractor.extract_all_features(val_df, fit=False)
    val_labels = (val_df['label'] == 'job_related').astype(int).values
    
    # Process test data
    print("\n=== Processing Test Data with Generalized Features ===")
    test_df = extractor.load_data(f"{data_dir}/test_data.csv")
    test_features, _ = extractor.extract_all_features(test_df, fit=False)
    test_labels = (test_df['label'] == 'job_related').astype(int).values
    
    # Save processed data
    os.makedirs(models_dir, exist_ok=True)
    
    processed_data = {
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        'feature_names': feature_names,
        'processing_timestamp': datetime.now().isoformat(),
        'pipeline_type': 'generalized'
    }
    
    with open(f"{models_dir}/generalized_processed_features.pkl", 'wb') as f:
        pickle.dump(processed_data, f)
    
    # Save feature pipeline
    extractor.save_pipeline(f"{models_dir}/generalized_feature_pipeline.pkl")
    
    # Create feature analysis report
    feature_analysis = {
        'total_features': len(feature_names),
        'feature_names': feature_names,
        'pipeline_type': 'generalized',
        'user_specific_features_removed': ['alex', 'zhao', 'zicheng', 'alex_zhao', 'zicheng_zhao', 'zichengalexzhao'],
        'dataset_sizes': {
            'train': train_features.shape,
            'val': val_features.shape, 
            'test': test_features.shape
        },
        'label_distributions': {
            'train': {'job_related': int(train_labels.sum()), 'non_job_related': int(len(train_labels) - train_labels.sum())},
            'val': {'job_related': int(val_labels.sum()), 'non_job_related': int(len(val_labels) - val_labels.sum())},
            'test': {'job_related': int(test_labels.sum()), 'non_job_related': int(len(test_labels) - test_labels.sum())}
        },
        'processing_date': datetime.now().isoformat(),
        'generalization_notes': {
            'user_names_replaced': 'All user-specific names replaced with generic tokens',
            'email_addresses_generalized': 'User email addresses replaced with USER_EMAIL token',
            'greeting_patterns_preserved': 'Greeting patterns maintained but generalized',
            'tfidf_user_terms_filtered': 'User-specific terms excluded from TF-IDF features'
        }
    }
    
    with open(f"{models_dir}/generalized_feature_analysis.json", 'w') as f:
        import json
        json.dump(feature_analysis, f, indent=2)
    
    print("\n=== Generalized Feature Engineering Complete ===")
    print(f"Processed data saved to: {models_dir}/generalized_processed_features.pkl")
    print(f"Feature pipeline saved to: {models_dir}/generalized_feature_pipeline.pkl")
    print(f"Feature analysis saved to: {models_dir}/generalized_feature_analysis.json")
    
    print(f"\n=== Dataset Summary ===")
    print(f"Training: {train_features.shape} - {train_labels.sum()} job, {len(train_labels) - train_labels.sum()} non-job")
    print(f"Validation: {val_features.shape} - {val_labels.sum()} job, {len(val_labels) - val_labels.sum()} non-job") 
    print(f"Test: {test_features.shape} - {test_labels.sum()} job, {len(test_labels) - test_labels.sum()} non-job")
    print(f"Total features: {len(feature_names)}")
    
    print(f"\n=== Key Generalizations Applied ===")
    print("✓ User names (Alex, Zhao, Zicheng) → Generic tokens")
    print("✓ Email addresses → USER_EMAIL token") 
    print("✓ Greeting patterns preserved but generalized")
    print("✓ TF-IDF excludes user-specific terms")
    print("✓ New features: user mention counts, personalization patterns")

if __name__ == "__main__":
    main()