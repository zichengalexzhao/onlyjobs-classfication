#!/usr/bin/env python3
"""
Final Optimized Gmail Test
Tests the optimized model (Step 1 + Step 2) on real Gmail data.
Uses the working incremental approach.
"""

import os
import sys
import json
import base64
import re
import numpy as np
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Add the original ml_models directory to path
sys.path.append('/Users/zichengzhao/Downloads/job-app-tracker/ml_models')

from feature_engineering import EmailFeatureExtractor

class FinalOptimizedExtractor:
    def __init__(self):
        self.base_extractor = EmailFeatureExtractor()
        
        # Step 1: Job Patterns (High Impact) ⭐
        self.job_patterns = [
            r'thank you for applying', r'thanks for applying', r'application.*received',
            r'we.*received.*application', r'interview.*scheduled?', r'schedule.*interview',
            r'job.*opportunity', r'position.*available', r'hr.*department',
            r'human resources', r'talent.*acquisition', r'recruiting.*team',
            r'hiring.*manager', r'offer.*letter', r'job.*offer',
            r'technical.*assessment', r'coding.*challenge', r'take.*home.*test',
            r'unfortunately.*not.*proceed', r'decided.*other.*candidate'
        ]
        
        # Step 2: Domain Analysis (Medium Impact) 🏢
        self.job_domains = [
            'greenhouse.io', 'lever.co', 'workday.com', 'jobvite.com',
            'careers.', 'jobs.', 'talent.', 'recruiting.'
        ]
        
        self.service_domains = [
            'amazon.com', 'netflix.com', 'uber.com', 'zelle.com',
            'chase.com', 'poshmark.com', 'paypal.com'
        ]
    
    def extract_optimized_features(self, emails):
        """Extract Step 1 + Step 2 features only"""
        
        # Base features
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        base_features, _ = self.base_extractor.extract_all_features(emails)
        sys.stdout = old_stdout
        
        # Step 1: Job pattern features
        step1_features = []
        for email in emails:
            content = email.get('full_content', '').lower()
            
            pattern_count = sum(1 for pattern in self.job_patterns if re.search(pattern, content))
            
            features = [
                pattern_count,  # Total pattern count
                1 if re.search(r'thank you for applying|application.*received', content) else 0,
                1 if re.search(r'interview.*scheduled?|schedule.*interview', content) else 0,
                1 if re.search(r'job.*opportunity|position.*available', content) else 0,
                1 if re.search(r'hr|recruiting|hiring.*manager', content) else 0,
                1 if re.search(r'offer.*letter|job.*offer', content) else 0,
                1 if re.search(r'assessment|coding.*challenge', content) else 0,
                1 if re.search(r'unfortunately.*not.*proceed|decided.*other.*candidate', content) else 0,
                (pattern_count / max(len(content.split()), 1)) * 100  # Pattern density
            ]
            step1_features.append(features)
        
        step1_features = np.array(step1_features)
        
        # Step 2: Domain features
        step2_features = []
        for email in emails:
            content = email.get('full_content', '')
            
            domains = re.findall(r'https?://([^/\\s]+)', content, re.IGNORECASE)
            domains.extend(re.findall(r'@([^.\\s]+\\.[^.\\s]+)', content, re.IGNORECASE))
            
            job_domain_count = 0
            service_domain_count = 0
            domain_score = 0
            
            for domain in domains:
                domain = domain.lower()
                
                if any(job_domain in domain for job_domain in self.job_domains):
                    job_domain_count += 1
                    domain_score += 3
                
                if any(service_domain in domain for service_domain in self.service_domains):
                    service_domain_count += 1
                    domain_score -= 2
                    
                if 'careers.' in domain or 'jobs.' in domain:
                    domain_score += 5
            
            has_noreply = 1 if re.search(r'no-?reply|noreply', content, re.IGNORECASE) else 0
            
            features = [
                1 if job_domain_count > 0 else 0,    # Has job domain
                1 if service_domain_count > 0 else 0, # Has service domain
                job_domain_count,                     # Job domain count
                service_domain_count,                 # Service domain count
                domain_score,                         # Overall domain score
                has_noreply                          # Has no-reply sender
            ]
            step2_features.append(features)
        
        step2_features = np.array(step2_features)
        
        # Combine all features
        combined_features = np.hstack([base_features, step1_features, step2_features])
        
        return combined_features

class FinalOptimizedGmailTester:
    def __init__(self, credentials_file, gmail_address):
        self.credentials_file = credentials_file
        self.gmail_address = gmail_address
        self.scopes = ['https://www.googleapis.com/auth/gmail.readonly']
        self.service = None
        self.extractor = FinalOptimizedExtractor()
        self.model = None
        
    def setup_gmail_auth(self):
        """Set up Gmail authentication"""
        print("🔐 Setting up Gmail authentication...")
        
        creds = None
        token_path = 'gmail_token.json'
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.scopes)
            
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except:
                    creds = None
                    
            if not creds:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, self.scopes)
                creds = flow.run_local_server(port=0)
            
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
                
        self.service = build('gmail', 'v1', credentials=creds)
        print("✅ Gmail authentication successful!")
        return True
    
    def train_final_model(self):
        """Train the final optimized model"""
        print("🎯 Training Final Optimized Model (Step 1 + Step 2)")
        
        # Load training data
        train_path = "/Users/zichengzhao/Downloads/job-app-tracker/data/ml_training/train_data.json"
        val_path = "/Users/zichengzhao/Downloads/job-app-tracker/data/ml_training/val_data.json"
        
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        with open(val_path, 'r') as f:
            val_data = json.load(f)
        
        all_emails = train_data + val_data
        
        # Extract optimized features
        features = self.extractor.extract_optimized_features(all_emails)
        labels = [1 if email['label'] == 'job_related' else 0 for email in all_emails]
        
        # Train Random Forest with high recall
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            class_weight={0: 1.0, 1: 3.0},  # 3x penalty for missing job emails
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(features, labels)
        
        print(f"✅ Final model trained!")
        print(f"   Features: {features.shape[1]} (Base + Job Patterns + Domain Analysis)")
        print(f"   Training samples: {len(labels)}")
        print(f"   Class weights: 3:1 (heavily penalize missing job emails)")
        
        return True
    
    def get_email_content(self, message_id):
        """Get email content"""
        try:
            message = self.service.users().messages().get(userId='me', id=message_id, format='full').execute()
            
            headers = message['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            
            body = self.extract_message_body(message['payload'])
            
            return {
                'subject': subject,
                'body': body,
                'full_content': f"Subject: {subject}\\n\\n{body}"
            }
        except:
            return None
    
    def extract_message_body(self, payload):
        """Extract message body"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif part['mimeType'] == 'text/html' and 'data' in part['body'] and not body:
                    html_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    body += re.sub('<.*?>', '', html_content)
        else:
            if payload['mimeType'] == 'text/plain' and 'data' in payload['body']:
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
            elif payload['mimeType'] == 'text/html' and 'data' in payload['body']:
                html_content = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
                body = re.sub('<.*?>', '', html_content)
        
        return body.strip()
    
    def fetch_recent_emails(self, max_results=20):
        """Fetch recent emails"""
        print(f"📬 Fetching {max_results} recent emails...")
        
        try:
            results = self.service.users().messages().list(userId='me', maxResults=max_results).execute()
            messages = results.get('messages', [])
            
            emails = []
            for i, message in enumerate(messages, 1):
                print(f"   Fetching email {i}/{len(messages)}...", end='\\r')
                email_content = self.get_email_content(message['id'])
                if email_content:
                    emails.append(email_content)
            
            print(f"\\n✅ Successfully fetched {len(emails)} emails")
            return emails
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def classify_and_display(self, emails):
        """Classify emails and display results"""
        print(f"🎯 Classifying {len(emails)} emails with final optimized model...")
        
        results = []
        
        for i, email in enumerate(emails, 1):
            email_obj = {
                'full_content': email['full_content'],
                'snippet': email['body'][:200],
                'label': 'unknown'
            }
            
            # Extract features and predict
            features = self.extractor.extract_optimized_features([email_obj])
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            predicted_label = 'JOB-RELATED' if prediction == 1 else 'NON-JOB-RELATED'
            
            results.append({
                'email_num': i,
                'subject': email['subject'],
                'prediction': predicted_label,
                'job_prob': probabilities[1],
                'confidence': max(probabilities),
                'body_preview': email['body'][:120] + '...' if len(email['body']) > 120 else email['body']
            })
        
        # Display results
        print("\\n🎯 Final Optimized Classification Results:")
        print("=" * 80)
        
        for result in results:
            print(f"\\n📧 Email {result['email_num']:2d}: {result['subject'][:55]}{'...' if len(result['subject']) > 55 else ''}")
            print(f"    → {result['prediction']} ({result['job_prob']:.1%} job probability)")
            print(f"    → Confidence: {result['confidence']:.1%}")
            print(f"    → Preview: {result['body_preview']}")
        
        # Summary
        job_count = sum(1 for r in results if r['prediction'] == 'JOB-RELATED')
        avg_job_prob = sum(r['job_prob'] for r in results) / len(results)
        
        print(f"\\n📊 Final Model Summary:")
        print(f"   Total emails: {len(results)}")
        print(f"   Job-related: {job_count}/{len(results)}")
        print(f"   Non-job-related: {len(results)-job_count}/{len(results)}")
        print(f"   Average job probability: {avg_job_prob:.1%}")
        
        print(f"\\n🏆 Model Configuration:")
        print(f"   ✅ Step 1: Job Pattern Detection (high impact)")
        print(f"   ✅ Step 2: Domain Analysis (medium impact)")
        print(f"   ❌ Step 3: Subject Analysis (dropped - low impact)")
        print(f"   🎯 Optimization: High Recall (3:1 class weights)")
        
        return results
    
    def run_final_test(self, num_emails=20):
        """Run the final optimized test"""
        print("🚀 Final Optimized Gmail Model Test")
        print("=" * 60)
        print(f"Gmail: {self.gmail_address}")
        print(f"Model: Final Optimized (Step 1 + Step 2 only)")
        print(f"Goal: Minimize false negatives (missed job emails)")
        
        if not self.setup_gmail_auth():
            return False
        
        if not self.train_final_model():
            return False
        
        emails = self.fetch_recent_emails(num_emails)
        if not emails:
            return False
        
        results = self.classify_and_display(emails)
        
        print(f"\\n🎉 Final test completed!")
        print("\\n💡 This is the production-ready model with:")
        print("   • Best performing features only")
        print("   • High recall optimization") 
        print("   • Minimal complexity")
        print("\\n   Ready to sync to GitHub repository!")
        
        return True

def main():
    credentials_file = "gmail_credentials.json"
    gmail_address = "zichengalexzhao@gmail.com"
    
    tester = FinalOptimizedGmailTester(credentials_file, gmail_address)
    tester.run_final_test(20)

if __name__ == "__main__":
    main()