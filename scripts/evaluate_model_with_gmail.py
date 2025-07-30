#!/usr/bin/env python3
"""
Model Evaluation Script with Gmail API
Fetches 100 emails from Gmail, runs ML predictions, and creates Excel file for manual review.
"""

import os
import sys
import json
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import argparse

# Gmail API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
import email
from email.mime.text import MIMEText

# Add src to path for our ML components
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feature_engineering.feature_pipeline import EmailFeatureExtractor

class GmailModelEvaluator:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.service = None
        self.model = None
        self.feature_extractor = None
        
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
                # Try running with a specific port and timeout
                try:
                    creds = flow.run_local_server(port=8080, timeout_seconds=300)
                except Exception as e:
                    print(f"❌ OAuth timeout or error: {e}")
                    print("💡 Please try running the script again when you have a stable internet connection.")
                    raise
            
            # Save credentials for next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
        print("✅ Gmail API authenticated successfully!")
        
    def load_ml_model(self):
        """Load the trained ML model and feature extractor"""
        print("🤖 Loading ML model...")
        
        # Load model
        model_path = "data/models/best_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load feature extractor
        self.feature_extractor = EmailFeatureExtractor()
        extractor_path = "data/models/feature_extractors.pkl"
        if not self.feature_extractor.load_feature_extractors(extractor_path):
            raise FileNotFoundError(f"Feature extractors not found: {extractor_path}")
        
        print("✅ ML model loaded successfully!")
    
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
                elif part['mimeType'] == 'text/html':
                    # If no plain text, use HTML as fallback
                    if not body:
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
    
    def fetch_emails(self, max_emails=100):
        """Fetch recent emails from Gmail"""
        print(f"📧 Fetching {max_emails} recent emails...")
        
        # Get list of message IDs
        results = self.service.users().messages().list(
            userId='me', 
            maxResults=max_emails,
            q='has:nouserlabels'  # Get emails without user labels for more variety
        ).execute()
        
        messages = results.get('messages', [])
        print(f"Found {len(messages)} messages to process...")
        
        emails = []
        for i, message in enumerate(messages):
            print(f"Processing email {i+1}/{len(messages)}...", end='\\r')
            
            email_content = self.get_email_content(message['id'])
            if email_content:
                emails.append(email_content)
        
        print(f"\\n✅ Successfully fetched {len(emails)} emails!")
        return emails
    
    def predict_emails(self, emails):
        """Run ML predictions on emails"""
        print("🔮 Running ML predictions...")
        
        predictions = []
        for i, email in enumerate(emails):
            print(f"Predicting email {i+1}/{len(emails)}...", end='\\r')
            
            # Create email object for feature extraction
            email_obj = {
                'full_content': email['full_content'],
                'snippet': email['full_content'][:200],
                'label': 'unknown'
            }
            
            try:
                # Extract features
                features, _ = self.feature_extractor.extract_all_features([email_obj])
                
                # Make prediction
                prediction = self.model.predict(features)[0]
                probabilities = self.model.predict_proba(features)[0]
                
                predictions.append({
                    'is_job_related': bool(prediction),
                    'confidence': float(max(probabilities)),
                    'job_probability': float(probabilities[1]),
                    'non_job_probability': float(probabilities[0])
                })
                
            except Exception as e:
                print(f"\\n❌ Error predicting email {i+1}: {e}")
                predictions.append({
                    'is_job_related': False,
                    'confidence': 0.0,
                    'job_probability': 0.0,
                    'non_job_probability': 1.0
                })
        
        print(f"\\n✅ Completed {len(predictions)} predictions!")
        return predictions
    
    def create_excel_evaluation(self, emails, predictions, output_file="evaluation_results.xlsx"):
        """Create Excel file with emails, predictions, and manual review column"""
        print(f"📊 Creating Excel evaluation file: {output_file}")
        
        # Prepare data for Excel
        data = []
        for email, pred in zip(emails, predictions):
            data.append({
                'Email ID': email['message_id'],
                'Date': email['date'],
                'Sender': email['sender'],
                'Subject': email['subject'],
                'Email Body': email['body'][:500] + '...' if len(email['body']) > 500 else email['body'],  # Truncate for readability
                'Full Content': email['full_content'],  # Full content for reference
                'ML Prediction': 'JOB-RELATED' if pred['is_job_related'] else 'NON-JOB-RELATED',
                'Confidence %': f"{pred['confidence']*100:.1f}%",
                'Job Probability': f"{pred['job_probability']*100:.1f}%",
                'Non-Job Probability': f"{pred['non_job_probability']*100:.1f}%",
                'Manual Review': '',  # Empty column for manual review
                'Is Prediction Correct?': ''  # Dropdown will be added
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create Excel file with formatting
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Email Evaluation', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Email Evaluation']
            
            # Add data validation (dropdown) for manual review
            from openpyxl.worksheet.datavalidation import DataValidation
            
            # Create dropdown for "Is Prediction Correct?" column (column L)
            dv = DataValidation(
                type="list",
                formula1='"Correct,Incorrect,Unsure"',
                allow_blank=True
            )
            dv.error = 'Your entry is not in the list'
            dv.errorTitle = 'Invalid Entry'
            dv.prompt = 'Please select: Correct, Incorrect, or Unsure'
            dv.promptTitle = 'Manual Review'
            
            # Add validation to the column (assuming max 1000 rows)
            worksheet.add_data_validation(dv)
            dv.add(f'L2:L{len(data)+1}')  # Column L, starting from row 2
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                # Set reasonable limits
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Freeze the first row
            worksheet.freeze_panes = 'A2'
        
        print(f"✅ Excel file created: {output_file}")
        print(f"📋 Summary:")
        print(f"   Total emails: {len(emails)}")
        print(f"   Predicted JOB-RELATED: {sum(1 for p in predictions if p['is_job_related'])}")
        print(f"   Predicted NON-JOB-RELATED: {sum(1 for p in predictions if not p['is_job_related'])}")
        print(f"   Average confidence: {sum(p['confidence'] for p in predictions)/len(predictions)*100:.1f}%")
        
        return output_file
    
    def run_evaluation(self, max_emails=100, output_file=None):
        """Run the complete evaluation pipeline"""
        print("🚀 Starting Gmail Model Evaluation")
        print("=" * 60)
        
        # Set default output file with timestamp
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"model_evaluation_{timestamp}.xlsx"
        
        try:
            # Step 1: Authenticate Gmail
            self.authenticate_gmail()
            
            # Step 2: Load ML model
            self.load_ml_model()
            
            # Step 3: Fetch emails
            emails = self.fetch_emails(max_emails)
            
            if not emails:
                print("❌ No emails fetched. Exiting.")
                return None
            
            # Step 4: Run predictions
            predictions = self.predict_emails(emails)
            
            # Step 5: Create Excel file
            excel_file = self.create_excel_evaluation(emails, predictions, output_file)
            
            print(f"\\n🎉 Evaluation complete!")
            print(f"📄 Excel file: {excel_file}")
            print(f"📝 Please review the 'Is Prediction Correct?' column in Excel")
            
            return excel_file
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Evaluate ML model with Gmail emails")
    parser.add_argument(
        "--max-emails",
        type=int,
        default=100,
        help="Maximum number of emails to fetch (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output Excel file path (default: model_evaluation_TIMESTAMP.xlsx)"
    )
    
    args = parser.parse_args()
    
    evaluator = GmailModelEvaluator()
    evaluator.run_evaluation(args.max_emails, args.output)

if __name__ == "__main__":
    main()