#!/usr/bin/env python3
"""
Optimized Gmail Test Script
Tests the final optimized model on real Gmail data.
"""

import os
import sys
import json
import base64
import re
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.optimized_classifier import OptimizedJobClassifier

class OptimizedGmailTester:
    def __init__(self, credentials_file, gmail_address):
        self.credentials_file = credentials_file
        self.gmail_address = gmail_address
        self.scopes = ['https://www.googleapis.com/auth/gmail.readonly']
        self.service = None
        self.classifier = OptimizedJobClassifier()
        
    def setup_gmail_auth(self):
        """Set up Gmail API authentication"""
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
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        model_path = "data/models/optimized_job_classifier.pkl"
        
        if os.path.exists(model_path):
            print("📂 Loading existing optimized model...")
            
            import pickle
            with open(model_path, 'rb') as f:
                self.classifier.model = pickle.load(f)
            
            # Load feature extractors
            if self.classifier.feature_extractor.load_optimized_extractors():
                print("✅ Optimized model loaded successfully!")
                return True
            else:
                print("⚠️  Feature extractors missing, retraining...")
        
        print("🎯 Training optimized model...")
        return self.classifier.train_and_evaluate()
    
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
        except Exception as e:
            print(f"Error getting email {message_id}: {e}")
            return None
    
    def extract_message_body(self, payload):
        """Extract body text"""
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
    
    def classify_emails(self, emails):
        """Classify emails using optimized model"""
        print(f"🎯 Classifying {len(emails)} emails with optimized model...")
        
        results = []
        
        for i, email in enumerate(emails, 1):
            try:
                # Classify email
                classification = self.classifier.classify_email(email['full_content'])
                
                results.append({
                    'email_num': i,
                    'subject': email['subject'],
                    'prediction': 'JOB-RELATED' if classification['is_job_related'] else 'NON-JOB-RELATED',
                    'confidence': classification['confidence'],
                    'job_probability': classification['job_probability'],
                    'non_job_probability': classification['non_job_probability'],
                    'body_preview': email['body'][:150] + '...' if len(email['body']) > 150 else email['body']
                })
                
            except Exception as e:
                print(f"Error classifying email {i}: {e}")
                results.append({
                    'email_num': i,
                    'subject': email['subject'],
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'job_probability': 0.0,
                    'non_job_probability': 0.0,
                    'body_preview': 'Error processing email'
                })
        
        return results
    
    def display_optimized_results(self, results):
        """Display classification results"""
        print("\\n🎯 Optimized Model Classification Results:")
        print("=" * 80)
        
        for result in results:
            if result['prediction'] == 'ERROR':
                print(f"\\n❌ Email {result['email_num']:2d}: {result['subject'][:60]} - Processing Error")
                continue
                
            print(f"\\n📧 Email {result['email_num']:2d}: {result['subject'][:60]}{'...' if len(result['subject']) > 60 else ''}")
            print(f"    Prediction: {result['prediction']}")
            print(f"    Job probability: {result['job_probability']:.1%}")
            print(f"    Confidence: {result['confidence']:.1%}")
            print(f"    Preview: {result['body_preview']}")
        
        # Summary statistics
        valid_results = [r for r in results if r['prediction'] != 'ERROR']
        job_related = sum(1 for r in valid_results if r['prediction'] == 'JOB-RELATED')
        non_job_related = len(valid_results) - job_related
        avg_job_prob = sum(r['job_probability'] for r in valid_results) / len(valid_results) if valid_results else 0
        avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results) if valid_results else 0
        
        print(f"\\n📊 Optimized Model Summary:")
        print(f"   Total emails processed: {len(valid_results)}")
        print(f"   Job-related: {job_related}")
        print(f"   Non-job-related: {non_job_related}")
        print(f"   Average job probability: {avg_job_prob:.1%}")
        print(f"   Average confidence: {avg_confidence:.1%}")
        
        # Model info
        try:
            with open("data/models/optimized_model_info.json", 'r') as f:
                model_info = json.load(f)
            
            print(f"\\n🤖 Model Information:")
            print(f"   Name: {model_info.get('name', 'Unknown')}")
            print(f"   Features: {model_info.get('features', 'Unknown')}")
            print(f"   Optimization: {model_info.get('optimization', 'Unknown')}")
            print(f"   Training Recall: {model_info.get('performance', {}).get('recall', 0):.1%}")
            
        except:
            pass
    
    def run_optimized_test(self, num_emails=20):
        """Run optimized Gmail test"""
        print("🚀 Optimized Gmail Model Test")
        print("=" * 50)
        print(f"Gmail: {self.gmail_address}")
        print(f"Model: Optimized (Job Patterns + Domain Analysis)")
        print(f"Focus: High Recall - Minimize Missed Job Emails")
        
        if not self.setup_gmail_auth():
            return False
        
        if not self.load_or_train_model():
            return False
        
        emails = self.fetch_recent_emails(num_emails)
        if not emails:
            return False
        
        results = self.classify_emails(emails)
        self.display_optimized_results(results)
        
        print(f"\\n🎉 Optimized test completed!")
        print("\\n💡 This model uses the best-performing features:")
        print("   ✅ Job Pattern Detection (high impact)")
        print("   ✅ Domain Analysis (medium impact)")
        print("   ❌ Subject Analysis (dropped - low impact)")
        print("\\n   How do the results look compared to previous tests?")
        
        return True

def main():
    credentials_file = "../gmail_credentials.json"
    gmail_address = "zichengalexzhao@gmail.com"
    
    # Change to parent directory for proper paths
    os.chdir(Path(__file__).parent.parent)
    
    tester = OptimizedGmailTester(credentials_file, gmail_address)
    tester.run_optimized_test(20)

if __name__ == "__main__":
    main()