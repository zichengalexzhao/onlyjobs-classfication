#!/usr/bin/env python3
"""
Optimized Feature Pipeline
Uses Step 1 (Job Patterns) + Step 2 (Domain Analysis) for best performance.
"""

import os
import sys
import re
import pickle
import numpy as np
from pathlib import Path

# Add the original ml_models directory to path
sys.path.append('/Users/zichengzhao/Downloads/job-app-tracker/ml_models')

# Import from the original location
try:
    from feature_engineering import EmailFeatureExtractor
except ImportError:
    sys.path.append('/Users/zichengzhao/Downloads/job-app-tracker/ml_models')
    from feature_engineering import EmailFeatureExtractor

class OptimizedJobFeatureExtractor:
    def __init__(self):
        self.base_extractor = EmailFeatureExtractor()
        
        # Strong job patterns (Step 1 - High Impact)
        self.job_patterns = [
            # Application confirmations
            r'thank you for applying',
            r'thanks for applying', 
            r'application.*received',
            r'we.*received.*application',
            r'receipt.*application',
            
            # Interview related
            r'interview.*scheduled?',
            r'schedule.*interview',
            r'phone.*interview',
            r'video.*interview',
            
            # Job opportunity language
            r'job.*opportunity',
            r'position.*available',
            r'hiring.*for',
            r'role.*available',
            
            # HR/Recruiting language
            r'hr.*department',
            r'human resources',
            r'talent.*acquisition',
            r'recruiting.*team',
            r'hiring.*manager',
            
            # Application process
            r'next.*step.*hiring',
            r'move.*forward.*process',
            r'application.*status',
            
            # Offer related
            r'offer.*letter',
            r'job.*offer',
            r'pleased.*offer',
            r'congratulations.*offer',
            
            # Assessment/Testing
            r'technical.*assessment',
            r'coding.*challenge',
            r'take.*home.*test',
            r'complete.*assessment',
            
            # Rejection patterns (still job-related)
            r'unfortunately.*not.*proceed',
            r'decided.*other.*candidate',
            r'not.*moving.*forward',
            r'position.*filled'
        ]
        
        # Job-related domains (Step 2 - Medium Impact)
        self.job_domains = [
            'greenhouse.io', 'lever.co', 'workday.com', 'successfactors.com',
            'icims.com', 'jobvite.com', 'bamboohr.com', 'applytojob.com',
            'careers.', 'jobs.', 'talent.', 'recruiting.',
            'hire.', 'employment.', 'work.', 'opportunity.'
        ]
        
        # Service/non-job domains
        self.service_domains = [
            'amazon.com', 'netflix.com', 'uber.com', 'lyft.com',
            'zelle.com', 'chase.com', 'paypal.com', 'venmo.com',
            'poshmark.com', 'airbnb.com', 'booking.com',
            'doordash.com', 'grubhub.com', 'postmates.com'
        ]
    
    def extract_job_pattern_features(self, emails):
        """Extract job pattern features (Step 1)"""
        features = []
        
        for email in emails:
            content = email.get('full_content', '').lower()
            
            pattern_features = {
                'strong_job_pattern_count': 0,
                'has_application_confirmation': 0,
                'has_interview_language': 0,
                'has_opportunity_language': 0,
                'has_hr_language': 0,
                'has_process_language': 0,
                'has_offer_language': 0,
                'has_assessment_language': 0,
                'has_rejection_language': 0,
                'job_pattern_density': 0
            }
            
            # Count all strong patterns
            pattern_count = 0
            for pattern in self.job_patterns:
                if re.search(pattern, content):
                    pattern_count += 1
            
            pattern_features['strong_job_pattern_count'] = pattern_count
            
            # Specific pattern categories
            if re.search(r'thank you for applying|thanks for applying|application.*received', content):
                pattern_features['has_application_confirmation'] = 1
                
            if re.search(r'interview.*scheduled?|schedule.*interview|phone.*interview', content):
                pattern_features['has_interview_language'] = 1
                
            if re.search(r'job.*opportunity|position.*available|role.*available', content):
                pattern_features['has_opportunity_language'] = 1
                
            if re.search(r'hr|human resources|talent.*acquisition|recruiting|hiring.*manager', content):
                pattern_features['has_hr_language'] = 1
                
            if re.search(r'next.*step|move.*forward.*process|application.*status', content):
                pattern_features['has_process_language'] = 1
                
            if re.search(r'offer.*letter|job.*offer|congratulations.*offer', content):
                pattern_features['has_offer_language'] = 1
                
            if re.search(r'assessment|coding.*challenge|take.*home.*test', content):
                pattern_features['has_assessment_language'] = 1
                
            if re.search(r'unfortunately.*not.*proceed|decided.*other.*candidate|not.*moving.*forward', content):
                pattern_features['has_rejection_language'] = 1
            
            # Pattern density (patterns per 100 words)
            word_count = len(content.split())
            if word_count > 0:
                pattern_features['job_pattern_density'] = (pattern_count / word_count) * 100
            
            features.append(list(pattern_features.values()))
        
        return np.array(features)
    
    def extract_domain_features(self, emails):
        """Extract domain analysis features (Step 2)"""
        features = []
        
        for email in emails:
            content = email.get('full_content', '')
            
            domain_features = {
                'has_job_domain': 0,
                'has_service_domain': 0,
                'job_domain_count': 0,
                'service_domain_count': 0,
                'domain_job_score': 0,
                'has_careers_subdomain': 0,
                'has_noreply_sender': 0
            }
            
            # Extract all domains from email
            domains = re.findall(r'https?://([^/\\s]+)', content, re.IGNORECASE)
            domains.extend(re.findall(r'@([^.\\s]+\\.[^.\\s]+)', content, re.IGNORECASE))
            
            job_domain_matches = 0
            service_domain_matches = 0
            
            for domain in domains:
                domain = domain.lower()
                
                # Check for job domains
                for job_domain in self.job_domains:
                    if job_domain in domain:
                        domain_features['has_job_domain'] = 1
                        job_domain_matches += 1
                        domain_features['domain_job_score'] += 3
                        break
                
                # Check for service domains
                for service_domain in self.service_domains:
                    if service_domain in domain:
                        domain_features['has_service_domain'] = 1
                        service_domain_matches += 1
                        domain_features['domain_job_score'] -= 2
                        break
                
                # Careers subdomain
                if 'careers.' in domain or 'jobs.' in domain:
                    domain_features['has_careers_subdomain'] = 1
                    domain_features['domain_job_score'] += 5
            
            domain_features['job_domain_count'] = job_domain_matches
            domain_features['service_domain_count'] = service_domain_matches
            
            # No-reply patterns (common in automated job emails)
            if re.search(r'no-?reply|noreply|donotreply', content, re.IGNORECASE):
                domain_features['has_noreply_sender'] = 1
                domain_features['domain_job_score'] += 1
            
            features.append(list(domain_features.values()))
        
        return np.array(features)
    
    def extract_optimized_features(self, emails):
        """Extract optimized features (Step 1 + Step 2 only)"""
        print("🎯 Optimized Feature Engineering Pipeline")
        print("=" * 50)
        
        # Extract base features
        print("📊 Extracting base TF-IDF and keyword features...")
        import sys
        from io import StringIO
        
        # Suppress base extractor output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        base_features, _ = self.base_extractor.extract_all_features(emails)
        sys.stdout = old_stdout
        
        # Extract optimized features
        print("🎯 Extracting job pattern features (Step 1)...")
        job_pattern_features = self.extract_job_pattern_features(emails)
        
        print("🏢 Extracting domain analysis features (Step 2)...")
        domain_features = self.extract_domain_features(emails)
        
        # Combine all features
        print("🔗 Combining optimized features...")
        combined_features = np.hstack([
            base_features,
            job_pattern_features,
            domain_features
        ])
        
        print(f"✅ Optimized feature extraction complete!")
        print(f"   Total features: {combined_features.shape[1]}")
        print(f"   Base features: {base_features.shape[1]}")
        print(f"   Job pattern features: {job_pattern_features.shape[1]}")
        print(f"   Domain features: {domain_features.shape[1]}")
        
        return combined_features
    
    def save_optimized_extractors(self, filepath="data/models/optimized_extractors.pkl"):
        """Save the optimized feature extractors"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save base extractor and patterns
        self.base_extractor.save_feature_extractors("data/models/base_extractors.pkl")
        
        extractors = {
            'job_patterns': self.job_patterns,
            'job_domains': self.job_domains,
            'service_domains': self.service_domains
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(extractors, f)
        
        print(f"💾 Optimized extractors saved to: {filepath}")
    
    def load_optimized_extractors(self, filepath="data/models/optimized_extractors.pkl"):
        """Load the optimized feature extractors"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                extractors = pickle.load(f)
            
            self.job_patterns = extractors['job_patterns']
            self.job_domains = extractors['job_domains']
            self.service_domains = extractors['service_domains']
            
            # Load base extractor
            self.base_extractor.load_feature_extractors("data/models/base_extractors.pkl")
            
            print(f"📂 Optimized extractors loaded from: {filepath}")
            return True
        else:
            print(f"❌ Optimized extractors not found at: {filepath}")
            return False

def main():
    """Test optimized feature extraction"""
    print("🚀 Testing Optimized Feature Pipeline")
    
    # Load sample data
    train_path = "/Users/zichengzhao/Downloads/job-app-tracker/data/ml_training/train_data.json"
    
    if os.path.exists(train_path):
        import json
        with open(train_path, 'r') as f:
            sample_data = json.load(f)[:50]  # Test with 50 samples
        
        extractor = OptimizedJobFeatureExtractor()
        features = extractor.extract_optimized_features(sample_data)
        
        print(f"\\n✅ Optimized extraction test successful!")
        print(f"   Sample size: {len(sample_data)} emails")
        print(f"   Feature matrix shape: {features.shape}")
        
        extractor.save_optimized_extractors()
        
    else:
        print("❌ No training data found for testing")

if __name__ == "__main__":
    main()