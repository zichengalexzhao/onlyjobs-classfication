#!/usr/bin/env python3
"""
Create Sample Evaluation File
Creates a sample Excel file with demo emails and predictions for testing the evaluation workflow.
"""

import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime
import pickle

# Add src to path for our ML components
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feature_engineering.feature_pipeline import EmailFeatureExtractor

def create_sample_data():
    """Create sample email data for testing"""
    sample_emails = [
        {
            'message_id': 'msg001',
            'date': '2024-01-15',
            'sender': 'careers@techcorp.com',
            'subject': 'Thank you for your application - Software Engineer',
            'body': 'Dear John, Thank you for applying to the Software Engineer position at TechCorp. We have received your application and our hiring team will review it carefully. If your qualifications match our requirements, we will contact you within the next two weeks to discuss the next steps in our interview process. Best regards, TechCorp Recruiting Team',
            'full_content': 'Subject: Thank you for your application - Software Engineer\\n\\nFrom: careers@techcorp.com\\n\\nDear John, Thank you for applying to the Software Engineer position at TechCorp. We have received your application and our hiring team will review it carefully. If your qualifications match our requirements, we will contact you within the next two weeks to discuss the next steps in our interview process. Best regards, TechCorp Recruiting Team'
        },
        {
            'message_id': 'msg002',
            'date': '2024-01-16',
            'sender': 'noreply@amazon.com',
            'subject': 'Your Amazon order has shipped',
            'body': 'Hello, Your order #123456789 has been shipped and is on its way. Track your package using the link below. Expected delivery: January 18, 2024. Thank you for shopping with Amazon.',
            'full_content': 'Subject: Your Amazon order has shipped\\n\\nFrom: noreply@amazon.com\\n\\nHello, Your order #123456789 has been shipped and is on its way. Track your package using the link below. Expected delivery: January 18, 2024. Thank you for shopping with Amazon.'
        },
        {
            'message_id': 'msg003',
            'date': '2024-01-17',
            'sender': 'talent@startup.io',
            'subject': 'Interview scheduled for Data Scientist role',
            'body': 'Hi Sarah, We would like to schedule a phone interview for the Data Scientist position you applied for. Are you available for a 30-minute call this Thursday at 2 PM PST? Please confirm your availability. Looking forward to speaking with you. Best, Startup Talent Team',
            'full_content': 'Subject: Interview scheduled for Data Scientist role\\n\\nFrom: talent@startup.io\\n\\nHi Sarah, We would like to schedule a phone interview for the Data Scientist position you applied for. Are you available for a 30-minute call this Thursday at 2 PM PST? Please confirm your availability. Looking forward to speaking with you. Best, Startup Talent Team'
        },
        {
            'message_id': 'msg004',
            'date': '2024-01-18',
            'sender': 'billing@netflix.com',
            'subject': 'Your Netflix payment was processed',
            'body': 'Your monthly Netflix subscription payment of $15.99 has been successfully processed. Your next billing date is February 18, 2024. Enjoy watching! Netflix Team',
            'full_content': 'Subject: Your Netflix payment was processed\\n\\nFrom: billing@netflix.com\\n\\nYour monthly Netflix subscription payment of $15.99 has been successfully processed. Your next billing date is February 18, 2024. Enjoy watching! Netflix Team'
        },
        {
            'message_id': 'msg005',
            'date': '2024-01-19',
            'sender': 'hr@bigcorp.com',
            'subject': 'Job offer - Senior Developer Position',
            'body': 'Congratulations! We are pleased to offer you the Senior Developer position at BigCorp. The salary is $120,000 per year with excellent benefits. Please review the attached offer letter and let us know your decision by January 25, 2024. We are excited about the possibility of you joining our team. HR Department',
            'full_content': 'Subject: Job offer - Senior Developer Position\\n\\nFrom: hr@bigcorp.com\\n\\nCongratulations! We are pleased to offer you the Senior Developer position at BigCorp. The salary is $120,000 per year with excellent benefits. Please review the attached offer letter and let us know your decision by January 25, 2024. We are excited about the possibility of you joining our team. HR Department'
        },
        {
            'message_id': 'msg006',
            'date': '2024-01-20',
            'sender': 'support@uber.com',
            'subject': 'Your ride receipt - $12.45',
            'body': 'Thanks for riding with Uber! Your trip on January 20, 2024 from Downtown to Airport cost $12.45. Payment was charged to your card ending in 1234. Rate your driver and share your experience. Uber',
            'full_content': 'Subject: Your ride receipt - $12.45\\n\\nFrom: support@uber.com\\n\\nThanks for riding with Uber! Your trip on January 20, 2024 from Downtown to Airport cost $12.45. Payment was charged to your card ending in 1234. Rate your driver and share your experience. Uber'
        },
        {
            'message_id': 'msg007',
            'date': '2024-01-21',
            'sender': 'recruiting@fintech.com',
            'subject': 'Technical assessment for Backend Engineer role',
            'body': 'Hello, Thank you for your interest in the Backend Engineer position. Please complete the attached technical assessment within 48 hours. The assessment covers algorithms, system design, and database concepts. Submit your solutions via the provided link. Good luck! Fintech Recruiting',
            'full_content': 'Subject: Technical assessment for Backend Engineer role\\n\\nFrom: recruiting@fintech.com\\n\\nHello, Thank you for your interest in the Backend Engineer position. Please complete the attached technical assessment within 48 hours. The assessment covers algorithms, system design, and database concepts. Submit your solutions via the provided link. Good luck! Fintech Recruiting'
        },
        {
            'message_id': 'msg008',
            'date': '2024-01-22',
            'sender': 'alerts@chase.com',
            'subject': 'Chase account alert - Transaction processed',
            'body': 'A transaction of $89.99 was processed on your Chase account ending in 5678. Transaction: Online purchase at BestBuy.com. If you did not authorize this transaction, please contact us immediately. Chase Security',
            'full_content': 'Subject: Chase account alert - Transaction processed\\n\\nFrom: alerts@chase.com\\n\\nA transaction of $89.99 was processed on your Chase account ending in 5678. Transaction: Online purchase at BestBuy.com. If you did not authorize this transaction, please contact us immediately. Chase Security'
        },
        {
            'message_id': 'msg009',
            'date': '2024-01-23',
            'sender': 'careers@google.com',
            'subject': 'Application status update',
            'body': 'Thank you for your interest in Google. After careful consideration, we have decided to move forward with other candidates for the Software Engineer position. We encourage you to apply for future opportunities that match your skills. Best wishes in your job search. Google Careers',
            'full_content': 'Subject: Application status update\\n\\nFrom: careers@google.com\\n\\nThank you for your interest in Google. After careful consideration, we have decided to move forward with other candidates for the Software Engineer position. We encourage you to apply for future opportunities that match your skills. Best wishes in your job search. Google Careers'
        },
        {
            'message_id': 'msg010',
            'date': '2024-01-24',
            'sender': 'updates@linkedin.com',
            'subject': 'You have 5 new connections',
            'body': 'You have 5 new connections this week! Check out their profiles and start networking. New connections can lead to great opportunities. Update your profile to attract more connections. LinkedIn Team',
            'full_content': 'Subject: You have 5 new connections\\n\\nFrom: updates@linkedin.com\\n\\nYou have 5 new connections this week! Check out their profiles and start networking. New connections can lead to great opportunities. Update your profile to attract more connections. LinkedIn Team'
        }
    ]
    
    return sample_emails

def load_ml_model():
    """Load the trained ML model and feature extractor"""
    print("🤖 Loading ML model...")
    
    # Load model
    model_path = "data/models/best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature extractor
    feature_extractor = EmailFeatureExtractor()
    extractor_path = "data/models/feature_extractors.pkl"
    if not feature_extractor.load_feature_extractors(extractor_path):
        raise FileNotFoundError(f"Feature extractors not found: {extractor_path}")
    
    print("✅ ML model loaded successfully!")
    return model, feature_extractor

def predict_emails(emails, model, feature_extractor):
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
            # Suppress output
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            # Extract features
            features, _ = feature_extractor.extract_all_features([email_obj])
            
            # Restore output
            sys.stdout = old_stdout
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
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

def create_excel_evaluation(emails, predictions, output_file="sample_evaluation.xlsx"):
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
            'Email Body': email['body'][:300] + '...' if len(email['body']) > 300 else email['body'],
            'Full Content': email['full_content'],
            'ML Prediction': 'JOB-RELATED' if pred['is_job_related'] else 'NON-JOB-RELATED',
            'Confidence %': f"{pred['confidence']*100:.1f}%",
            'Job Probability': f"{pred['job_probability']*100:.1f}%",
            'Non-Job Probability': f"{pred['non_job_probability']*100:.1f}%",
            'Manual Review': '',
            'Is Prediction Correct?': ''
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
        
        # Add validation to the column
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
    return output_file

def main():
    print("🚀 Creating Sample Model Evaluation")
    print("=" * 50)
    
    try:
        # Create sample data
        print("📧 Creating sample email data...")
        emails = create_sample_data()
        print(f"Created {len(emails)} sample emails")
        
        # Load ML model
        model, feature_extractor = load_ml_model()
        
        # Run predictions
        predictions = predict_emails(emails, model, feature_extractor)
        
        # Create Excel file
        excel_file = create_excel_evaluation(emails, predictions)
        
        print(f"\\n🎉 Sample evaluation complete!")
        print(f"📄 Excel file: {excel_file}")
        print(f"📝 Open the file and use the dropdown in 'Is Prediction Correct?' column")
        print(f"📊 Summary:")
        print(f"   Total emails: {len(emails)}")
        print(f"   Predicted JOB-RELATED: {sum(1 for p in predictions if p['is_job_related'])}")
        print(f"   Predicted NON-JOB-RELATED: {sum(1 for p in predictions if not p['is_job_related'])}")
        print(f"   Average confidence: {sum(p['confidence'] for p in predictions)/len(predictions)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Sample evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()