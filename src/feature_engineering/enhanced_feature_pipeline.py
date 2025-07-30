#!/usr/bin/env python3
"""
Enhanced Feature Pipeline for Reduced False Negatives
Specifically designed to catch job-related emails with improved patterns.
"""

import os
import sys
import re
import pickle
import numpy as np
from pathlib import Path

# Import the base feature extractor
sys.path.append(str(Path(__file__).parent))
from feature_pipeline import EmailFeatureExtractor

class EnhancedJobFeatureExtractor(EmailFeatureExtractor):
    def __init__(self):
        super().__init__()
        
        # Enhanced job-related domain patterns (addresses 12/19 false negatives)
        self.job_domains = [
            # Workday systems (major source of false negatives)
            'myworkday.com',
            'workday.com',
            'workday.',
            
            # ATS and recruiting platforms
            'greenhouse.io',
            'lever.co',
            'successfactors.com',
            'icims.com',
            'jobvite.com',
            'bamboohr.com',
            'applytojob.com',
            'newtonsoftware.com',
            'echo.newtonsoftware.com',
            'workable.com',
            'smartrecruiters.com',
            
            # Career site subdomains
            'careers.',
            'jobs.',
            'talent.',
            'recruiting.',
            'hire.',
            'employment.',
            'work.',
            'opportunity.',
            
            # Company HR domains
            'hr.',
            'people.',
            'recruitment.',
            'talentacquisition.'
        ]
        
        # Enhanced job-related keywords (more comprehensive)
        self.enhanced_job_keywords = [
            # Application confirmations (addresses many false negatives)
            'application received',
            'thank you for applying',
            'thanks for applying',
            'application confirmation',
            'we received your application',
            'your application for',
            'application submitted',
            
            # Interview related
            'interview request',
            'interview scheduled',
            'schedule an interview',
            'phone interview',
            'video interview',
            'technical interview',
            'take home test',
            'coding challenge',
            'assessment',
            
            # Job opportunity language
            'job opportunity',
            'position available',
            'role available',
            'opening available',
            'we are hiring',
            'join our team',
            
            # Recruiting language
            'talent acquisition',
            'recruiting team',
            'hiring manager',
            'hr department',
            'human resources',
            
            # Application process
            'next steps',
            'move forward',
            'application status',
            'under review',
            'review process',
            
            # Offer related
            'job offer',
            'offer letter',
            'congratulations',
            'pleased to offer',
            
            # Rejection (still job-related)
            'other candidates',
            'not moving forward',
            'unfortunately',
            'decided to proceed',
            'position has been filled'
        ]
        
        # Sender patterns that indicate job emails
        self.job_sender_patterns = [
            r'hr@',
            r'careers@',
            r'jobs@',
            r'recruiting@',
            r'talent@',
            r'noreply.*workday',
            r'workday.*noreply',
            r'.*@.*workday.*',
            r'talent.*acquisition',
            r'human.*resources'
        ]
        
        # Subject patterns that strongly indicate job emails
        self.job_subject_patterns = [
            r'thank.*you.*for.*applying',
            r'application.*received',
            r'interview.*request',
            r'interview.*scheduled',
            r'job.*application',
            r'application.*for.*',
            r'take.*home.*test',
            r'coding.*challenge',
            r'technical.*assessment',
            r'.*analyst.*opening',
            r'.*engineer.*position',
            r'.*scientist.*role',
            r'we.*received.*your.*application'
        ]
    
    def extract_enhanced_job_features(self, emails):
        """Extract enhanced job-related features to reduce false negatives"""
        print("🎯 Extracting enhanced job detection features...")
        
        features = []
        
        for email in emails:
            content = email.get('full_content', '').lower()
            sender = email.get('sender', '').lower()
            subject = email.get('subject', '').lower()
            
            job_features = {
                # Domain-based features (highest impact)
                'has_workday_domain': 0,
                'has_job_ats_domain': 0,
                'has_careers_subdomain': 0,
                'job_domain_count': 0,
                
                # Sender-based features
                'has_hr_sender': 0,
                'has_recruiting_sender': 0,
                'has_noreply_workday_sender': 0,
                
                # Subject line features (very strong indicators)
                'subject_has_application_confirmation': 0,
                'subject_has_interview_language': 0,
                'subject_has_job_title': 0,
                'subject_job_pattern_count': 0,
                
                # Content-based features
                'has_application_confirmation_language': 0,
                'has_interview_scheduling_language': 0,
                'has_strong_job_keywords': 0,
                'enhanced_job_keyword_count': 0,
                'job_keyword_density': 0,
                
                # Combined indicators
                'overall_job_confidence_score': 0
            }
            
            # Extract all domains from email
            email_text = f"{sender} {subject} {content}"
            domains = re.findall(r'https?://([^/\s]+)', email_text)
            domains.extend(re.findall(r'@([^.\s]+\.[^.\s]+)', email_text))
            
            job_domain_matches = 0
            
            for domain in domains:
                domain = domain.lower()
                
                # Check for Workday domains (major improvement)
                if 'workday' in domain:
                    job_features['has_workday_domain'] = 1
                    job_domain_matches += 3  # High weight
                
                # Check for other job domains
                for job_domain in self.job_domains:
                    if job_domain in domain:
                        job_features['has_job_ats_domain'] = 1
                        job_domain_matches += 2
                        break
                
                # Check for career subdomains
                if domain.startswith('careers.') or domain.startswith('jobs.'):
                    job_features['has_careers_subdomain'] = 1
                    job_domain_matches += 2
            
            job_features['job_domain_count'] = job_domain_matches
            
            # Analyze sender patterns
            for sender_pattern in self.job_sender_patterns:
                if re.search(sender_pattern, sender):
                    if 'hr' in sender_pattern or 'recruiting' in sender_pattern:
                        job_features['has_hr_sender'] = 1
                    if 'workday' in sender_pattern:
                        job_features['has_noreply_workday_sender'] = 1
                    job_features['has_recruiting_sender'] = 1
                    break
            
            # Analyze subject patterns
            subject_job_matches = 0
            for subject_pattern in self.job_subject_patterns:
                if re.search(subject_pattern, subject):
                    subject_job_matches += 1
                    
                    if 'thank' in subject_pattern and 'applying' in subject_pattern:
                        job_features['subject_has_application_confirmation'] = 1
                    if 'interview' in subject_pattern:
                        job_features['subject_has_interview_language'] = 1
                    if 'analyst' in subject_pattern or 'engineer' in subject_pattern or 'scientist' in subject_pattern:
                        job_features['subject_has_job_title'] = 1
            
            job_features['subject_job_pattern_count'] = subject_job_matches
            
            # Content analysis
            enhanced_keyword_count = 0
            for keyword in self.enhanced_job_keywords:
                if keyword in content:
                    enhanced_keyword_count += 1
                    
                    if 'application' in keyword and ('received' in keyword or 'thank' in keyword):
                        job_features['has_application_confirmation_language'] = 1
                    if 'interview' in keyword or 'schedule' in keyword:
                        job_features['has_interview_scheduling_language'] = 1
            
            job_features['enhanced_job_keyword_count'] = enhanced_keyword_count
            job_features['has_strong_job_keywords'] = 1 if enhanced_keyword_count >= 2 else 0
            
            # Calculate keyword density
            word_count = len(content.split())
            if word_count > 0:
                job_features['job_keyword_density'] = enhanced_keyword_count / word_count * 100
            
            # Calculate overall confidence score (weighted combination)
            confidence_score = 0
            confidence_score += job_features['has_workday_domain'] * 30  # Very strong indicator
            confidence_score += job_features['has_job_ats_domain'] * 20
            confidence_score += job_features['has_careers_subdomain'] * 15
            confidence_score += job_features['has_hr_sender'] * 25
            confidence_score += job_features['subject_has_application_confirmation'] * 35
            confidence_score += job_features['subject_has_interview_language'] * 30
            confidence_score += job_features['has_application_confirmation_language'] * 25
            confidence_score += job_features['has_interview_scheduling_language'] * 20
            confidence_score += min(job_features['enhanced_job_keyword_count'] * 5, 20)
            
            job_features['overall_job_confidence_score'] = min(confidence_score, 100)
            
            features.append(list(job_features.values()))
        
        return np.array(features)
    
    def extract_all_features_enhanced(self, emails):
        """Extract all features including enhanced job detection features"""
        print("🚀 Enhanced Feature Engineering Pipeline")
        print("=" * 60)
        
        # Extract base features (suppress output)
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        base_features, feature_names = self.extract_all_features(emails)
        sys.stdout = old_stdout
        
        # Extract enhanced job features
        enhanced_job_features = self.extract_enhanced_job_features(emails)
        
        # Combine all features
        print("🔗 Combining base and enhanced features...")
        combined_features = np.hstack([base_features, enhanced_job_features])
        
        # Update feature names
        enhanced_feature_names = [
            'has_workday_domain', 'has_job_ats_domain', 'has_careers_subdomain', 'job_domain_count',
            'has_hr_sender', 'has_recruiting_sender', 'has_noreply_workday_sender',
            'subject_has_application_confirmation', 'subject_has_interview_language', 
            'subject_has_job_title', 'subject_job_pattern_count',
            'has_application_confirmation_language', 'has_interview_scheduling_language',
            'has_strong_job_keywords', 'enhanced_job_keyword_count', 'job_keyword_density',
            'overall_job_confidence_score'
        ]
        
        all_feature_names = feature_names + enhanced_feature_names
        
        print(f"✅ Enhanced feature extraction complete!")
        print(f"   Base features: {base_features.shape[1]}")
        print(f"   Enhanced job features: {enhanced_job_features.shape[1]}")
        print(f"   Total features: {combined_features.shape[1]}")
        
        return combined_features, all_feature_names
    
    def save_enhanced_extractors(self, filepath="data/models/enhanced_feature_extractors.pkl"):
        """Save the enhanced feature extractors"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save base extractor
        self.save_feature_extractors("data/models/base_feature_extractors.pkl")
        
        # Save enhanced patterns
        enhanced_patterns = {
            'job_domains': self.job_domains,
            'enhanced_job_keywords': self.enhanced_job_keywords,
            'job_sender_patterns': self.job_sender_patterns,
            'job_subject_patterns': self.job_subject_patterns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(enhanced_patterns, f)
        
        print(f"💾 Enhanced extractors saved to: {filepath}")
    
    def load_enhanced_extractors(self, filepath="data/models/enhanced_feature_extractors.pkl"):
        """Load the enhanced feature extractors"""
        # Load base extractors first
        if not self.load_feature_extractors("data/models/base_feature_extractors.pkl"):
            print("❌ Failed to load base extractors")
            return False
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                enhanced_patterns = pickle.load(f)
            
            self.job_domains = enhanced_patterns['job_domains']
            self.enhanced_job_keywords = enhanced_patterns['enhanced_job_keywords']
            self.job_sender_patterns = enhanced_patterns['job_sender_patterns']
            self.job_subject_patterns = enhanced_patterns['job_subject_patterns']
            
            print(f"📂 Enhanced extractors loaded from: {filepath}")
            return True
        else:
            print(f"❌ Enhanced extractors not found at: {filepath}")
            return False

def main():
    """Test enhanced feature extraction"""
    print("🚀 Testing Enhanced Feature Pipeline")
    
    # Sample test data based on false negatives we identified
    test_emails = [
        {
            'sender': 'workday-noreply centene <centene@myworkday.com>',
            'subject': 'Thanks for Applying to Data Analyst IV (HEDIS, SQL) - 1582861',
            'full_content': 'Thank you for applying to the Data Analyst position. We have received your application and will review it.',
            'label': 'job_related'
        },
        {
            'sender': 'IQVIA Global Talent Acquisition <iqvia@myworkday.com>',
            'subject': 'Thank You for Applying',
            'full_content': 'We have received your application for the Data Scientist position at IQVIA.',
            'label': 'job_related'
        },
        {
            'sender': 'Chase <no.reply.alerts@chase.com>',
            'subject': 'Your credit card payment is scheduled',
            'full_content': 'Your payment of $123.45 has been scheduled for processing.',
            'label': 'non_job_related'
        }
    ]
    
    extractor = EnhancedJobFeatureExtractor()
    features, feature_names = extractor.extract_all_features_enhanced(test_emails)
    
    print(f"\n✅ Test completed!")
    print(f"   Sample emails: {len(test_emails)}")
    print(f"   Feature matrix shape: {features.shape}")
    print(f"   Total features: {len(feature_names)}")
    
    # Show some key enhanced features for the job-related emails
    print(f"\n🔍 Enhanced Job Features for Sample Emails:")
    enhanced_start_idx = features.shape[1] - 17  # Last 17 are enhanced features
    
    for i, email in enumerate(test_emails):
        print(f"\nEmail {i+1}: {email['subject'][:50]}...")
        print(f"   Label: {email['label']}")
        enhanced_features = features[i, enhanced_start_idx:]
        print(f"   Workday Domain: {enhanced_features[0]}")
        print(f"   Job ATS Domain: {enhanced_features[1]}")
        print(f"   Application Confirmation Subject: {enhanced_features[7]}")
        print(f"   Overall Job Confidence Score: {enhanced_features[-1]}")

if __name__ == "__main__":
    main()