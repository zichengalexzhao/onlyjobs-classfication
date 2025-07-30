#!/usr/bin/env python3
"""
Large Dataset Collection Script
Collects 5000 job-related + 5000 non-job-related emails for robust model training.
"""

import os
import sys
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Gmail API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64

# OpenAI for automated labeling
import openai

class LargeDatasetCollector:
    def __init__(self, openai_api_key=None, max_budget=30.0):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.service = None
        self.openai_api_key = openai_api_key
        self.max_budget = max_budget
        self.cost_per_classification = 0.003  # Estimated cost per OpenAI classification
        
        # Initialize OpenAI if API key provided
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Job-related search queries for targeted collection
        self.job_search_queries = [
            # Application confirmations
            'subject:"thank you for applying"',
            'subject:"application received"',
            'subject:"thanks for applying"',
            'subject:"we received your application"',
            
            # Workday systems (our major improvement area)
            'from:myworkday.com',
            'from:workday',
            'subject:"workday"',
            
            # Interview related
            'subject:"interview"',
            'subject:"schedule"',
            'subject:"phone call"',
            'subject:"zoom"',
            
            # Job opportunity
            'subject:"position"',
            'subject:"role"',
            'subject:"opportunity"',
            'subject:"opening"',
            
            # Recruiting
            'from:careers@',
            'from:jobs@',
            'from:recruiting@',
            'from:hr@',
            'from:talent@',
            
            # Assessment/testing
            'subject:"assessment"',
            'subject:"test"',
            'subject:"challenge"',
            'subject:"coding"',
            
            # Offer related
            'subject:"offer"',
            'subject:"congratulations"',
            
            # Rejection (still job-related)
            'subject:"unfortunately"',
            'subject:"other candidates"',
            'subject:"not moving forward"'
        ]
        
        # Non-job search queries for negative examples
        self.non_job_search_queries = [
            # Shopping and services
            'from:amazon.com',
            'from:netflix.com',
            'from:uber.com',
            'from:lyft.com',
            'from:doordash.com',
            'from:grubhub.com',
            
            # Financial
            'from:chase.com',
            'from:paypal.com',
            'from:venmo.com',
            'from:zelle.com',
            'subject:"payment"',
            'subject:"transaction"',
            'subject:"bill"',
            
            # Social media
            'from:facebook.com',
            'from:instagram.com',
            'from:twitter.com',
            'from:linkedin.com',
            
            # Housing/travel
            'from:airbnb.com',
            'from:booking.com',
            'subject:"reservation"',
            
            # Newsletters/marketing
            'subject:"newsletter"',
            'subject:"unsubscribe"',
            'subject:"sale"',
            'subject:"deal"',
            
            # System notifications
            'subject:"security"',
            'subject:"backup"',
            'subject:"update"'
        ]
    
    def authenticate_gmail(self):
        """Authenticate with Gmail API"""
        print("🔐 Authenticating with Gmail API...")
        
        creds = None
        token_path = "config/gmail_token.json"
        credentials_path = "config/gmail_credentials.json"
        
        # Load existing token
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
        
        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(f"Gmail credentials not found: {credentials_path}")
                
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, self.SCOPES)
                try:
                    creds = flow.run_local_server(port=8080, timeout_seconds=300)
                except Exception as e:
                    print(f"❌ OAuth timeout or error: {e}")
                    raise
            
            # Save credentials for next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
        print("✅ Gmail API authenticated successfully!")
    
    def search_emails_by_query(self, query, max_results=1000):
        """Search for emails using Gmail query syntax"""
        print(f"🔍 Searching: {query[:50]}...")
        
        try:
            # Add date filter to get recent emails (last 2 years)
            two_years_ago = datetime.now() - timedelta(days=730)
            date_filter = two_years_ago.strftime("after:%Y/%m/%d")
            full_query = f"{query} {date_filter}"
            
            results = self.service.users().messages().list(
                userId='me',
                q=full_query,
                maxResults=min(max_results, 500)  # Gmail API limit
            ).execute()
            
            messages = results.get('messages', [])
            print(f"   Found {len(messages)} messages")
            return [msg['id'] for msg in messages]
            
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            return []
    
    def get_email_content(self, message_id):
        """Get full email content from message ID"""
        try:
            message = self.service.users().messages().get(
                userId='me', id=message_id, format='full'
            ).execute()
            
            # Extract email content
            payload = message['payload']
            headers = payload.get('headers', [])
            
            # Get subject and sender
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
            
            # Get body content
            body = self.extract_body_content(payload)
            
            # Combine subject and body for classification
            full_content = f"Subject: {subject}\\n\\nFrom: {sender}\\n\\n{body}"
            
            return {
                'message_id': message_id,
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body,
                'full_content': full_content
            }
            
        except Exception as e:
            print(f"❌ Error fetching email {message_id}: {e}")
            return None
    
    def extract_body_content(self, payload):
        """Extract body content from email payload"""
        body = ""
        
        if 'parts' in payload:
            # Multi-part message
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                elif part['mimeType'] == 'text/html' and not body:
                    # If no plain text, use HTML as fallback
                    data = part['body']['data']
                    html_content = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                    # Simple HTML tag removal
                    import re
                    body = re.sub('<[^<]+?>', '', html_content)
        else:
            # Single part message
            if payload['body'].get('data'):
                data = payload['body']['data']
                body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        
        return body.strip()
    
    def classify_with_openai(self, email_content, expected_label=None):
        """Classify email using OpenAI (with budget tracking)"""
        if not self.openai_api_key:
            # If no OpenAI key, use simple heuristics based on expected label
            return expected_label if expected_label else 'unknown'
        
        try:
            prompt = f\"\"\"
Classify this email as either "job_related" or "non_job_related".

Email content:
{email_content[:1000]}  # Truncate to save tokens

Respond with only "job_related" or "non_job_related".
\"\"\"
            
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=10,
                temperature=0
            )
            
            classification = response.choices[0].text.strip().lower()
            
            if 'job_related' in classification:
                return 'job_related'
            elif 'non_job_related' in classification:
                return 'non_job_related'
            else:
                return expected_label if expected_label else 'unknown'
                
        except Exception as e:
            print(f"❌ OpenAI classification failed: {e}")
            return expected_label if expected_label else 'unknown'
    
    def collect_targeted_emails(self, search_queries, target_count, expected_label, label_name):
        """Collect emails using targeted search queries"""
        print(f"\\n📧 Collecting {target_count} {label_name} emails...")
        print("=" * 60)
        
        collected_emails = []
        seen_message_ids = set()
        
        # Shuffle queries for diversity
        queries = search_queries.copy()
        random.shuffle(queries)
        
        emails_per_query = max(target_count // len(queries), 100)
        
        for query in queries:
            if len(collected_emails) >= target_count:
                break
            
            # Search for emails
            message_ids = self.search_emails_by_query(query, emails_per_query)
            
            # Process each email
            for message_id in message_ids:
                if len(collected_emails) >= target_count:
                    break
                
                if message_id in seen_message_ids:
                    continue
                
                seen_message_ids.add(message_id)
                
                # Get email content
                email_content = self.get_email_content(message_id)
                if not email_content:
                    continue
                
                # Classify (use heuristics if no OpenAI key)
                if self.openai_api_key:
                    classification = self.classify_with_openai(
                        email_content['full_content'], 
                        expected_label
                    )
                else:
                    # Use expected label based on search query
                    classification = expected_label
                
                # Add to collection
                email_data = {
                    **email_content,
                    'label': classification,
                    'search_query': query,
                    'collection_method': 'targeted_search'
                }
                
                collected_emails.append(email_data)
                
                if len(collected_emails) % 100 == 0:
                    print(f"   Collected {len(collected_emails)}/{target_count} {label_name} emails...")
        
        print(f"✅ Collected {len(collected_emails)} {label_name} emails")
        return collected_emails
    
    def collect_large_dataset(self, job_target=5000, non_job_target=5000):
        """Collect large dataset with specified targets"""
        print("🚀 Large Dataset Collection")
        print("=" * 80)
        print(f"Target: {job_target} job-related + {non_job_target} non-job-related emails")
        print(f"Estimated cost: ${(job_target + non_job_target) * self.cost_per_classification:.2f}")
        print(f"Budget limit: ${self.max_budget:.2f}")
        
        if (job_target + non_job_target) * self.cost_per_classification > self.max_budget:
            print("⚠️  Estimated cost exceeds budget. Consider reducing targets or increasing budget.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return None, None
        
        # Authenticate Gmail
        self.authenticate_gmail()
        
        # Collect job-related emails
        job_emails = self.collect_targeted_emails(
            self.job_search_queries, 
            job_target, 
            'job_related', 
            'job-related'
        )
        
        # Collect non-job-related emails
        non_job_emails = self.collect_targeted_emails(
            self.non_job_search_queries, 
            non_job_target, 
            'non_job_related', 
            'non-job-related'
        )
        
        return job_emails, non_job_emails
    
    def save_dataset(self, job_emails, non_job_emails, output_dir="data/large_dataset"):
        """Save the collected dataset"""
        print(f"\\n💾 Saving Dataset")
        print("=" * 30)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine and shuffle
        all_emails = job_emails + non_job_emails
        random.shuffle(all_emails)
        
        # Split into train/val/test (70/20/10)
        total = len(all_emails)
        train_end = int(0.7 * total)
        val_end = int(0.9 * total)
        
        train_data = all_emails[:train_end]
        val_data = all_emails[train_end:val_end]
        test_data = all_emails[val_end:]
        
        # Save splits
        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in datasets.items():
            filepath = os.path.join(output_dir, f"{split_name}_data.json")
            with open(filepath, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"   {split_name}: {len(split_data)} emails saved to {filepath}")
        
        # Save collection metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'total_emails': len(all_emails),
            'job_related_emails': len(job_emails),
            'non_job_related_emails': len(non_job_emails),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'job_search_queries': self.job_search_queries,
            'non_job_search_queries': self.non_job_search_queries,
            'estimated_cost': len(all_emails) * self.cost_per_classification
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\\n✅ Dataset saved to {output_dir}")
        print(f"📊 Summary:")
        print(f"   Total emails: {len(all_emails)}")
        print(f"   Job-related: {len(job_emails)} ({len(job_emails)/len(all_emails)*100:.1f}%)")
        print(f"   Non-job-related: {len(non_job_emails)} ({len(non_job_emails)/len(all_emails)*100:.1f}%)")
        print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return output_dir

def main():
    parser = argparse.ArgumentParser(description="Collect large dataset for robust model training")
    parser.add_argument("--job-target", type=int, default=5000, help="Target number of job-related emails")
    parser.add_argument("--non-job-target", type=int, default=5000, help="Target number of non-job-related emails")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key for automated labeling")
    parser.add_argument("--budget", type=float, default=30.0, help="Maximum budget for OpenAI costs")
    parser.add_argument("--output-dir", type=str, default="data/large_dataset", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = LargeDatasetCollector(
        openai_api_key=args.openai_key,
        max_budget=args.budget
    )
    
    try:
        # Collect dataset
        job_emails, non_job_emails = collector.collect_large_dataset(
            args.job_target, 
            args.non_job_target
        )
        
        if job_emails is not None and non_job_emails is not None:
            # Save dataset
            output_dir = collector.save_dataset(job_emails, non_job_emails, args.output_dir)
            
            print(f"\\n🎉 Large dataset collection complete!")
            print(f"📁 Dataset saved to: {output_dir}")
            print(f"🚀 Ready for enhanced model training!")
        else:
            print("❌ Dataset collection cancelled or failed.")
    
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        raise

if __name__ == "__main__":
    main()