#!/usr/bin/env python3
"""
Example Integration of Production Email Classifier
Demonstrates how to integrate the email classifier into your application.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from models.production_email_classifier import ProductionEmailClassifier

def main():
    """Example of using the production email classifier in your application."""
    
    print("=== Email Classification Integration Example ===\n")
    
    # Initialize the classifier
    classifier = ProductionEmailClassifier()
    
    # Load the model
    print("Loading model...")
    if not classifier.load_model():
        print("❌ Failed to load model. Please ensure model files exist in data/models/")
        print("Required files:")
        print("  - data/models/generalized_email_classifier.pkl")
        print("  - data/models/generalized_feature_pipeline.pkl")
        return
    
    print("✅ Model loaded successfully!")
    print(f"   Model Version: {classifier.model_version}")
    print(f"   Accuracy: {classifier.accuracy*100:.1f}%")
    print(f"   Features: {classifier.feature_count}")
    
    # Health check
    health = classifier.health_check()
    print(f"\n🔍 Health Check: {health['status'].upper()}")
    
    if health['status'] != 'healthy':
        print(f"❌ Health check failed: {health.get('message', 'Unknown error')}")
        return
    
    # Test with example emails
    test_emails = [
        {
            "email_body": """
            Dear Candidate,
            
            Thank you for applying to our Software Engineer position at TechCorp. 
            We were impressed with your background and would like to schedule a 
            phone interview with our engineering team.
            
            Please let me know your availability for next week.
            
            Best regards,
            Sarah Johnson
            Engineering Manager
            """,
            "sender_email": "sarah.johnson@techcorp.com",
            "description": "Interview invitation email"
        },
        {
            "email_body": """
            Your Amazon order has been shipped!
            
            Order #123-4567890-1234567
            
            Track your package: [tracking link]
            
            Expected delivery: Tomorrow by 8PM
            """,
            "sender_email": "shipment-tracking@amazon.com", 
            "description": "Amazon shipping notification"
        },
        {
            "email_body": """
            Hello,
            
            We received your application for the Data Scientist role. 
            Unfortunately, we have decided to move forward with other candidates 
            whose experience more closely matches our current needs.
            
            Thank you for your interest in our company.
            
            Best of luck in your job search!
            """,
            "sender_email": "noreply@company.com",
            "description": "Job rejection email"
        },
        {
            "email_body": """
            Congratulations!
            
            We are pleased to extend an offer for the Senior Developer position.
            Your starting salary will be $120,000 with full benefits.
            
            Please review the attached offer letter and let us know by Friday.
            
            Welcome to the team!
            """,
            "sender_email": "hr@startup.com",
            "description": "Job offer email"
        }
    ]
    
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL EMAIL CLASSIFICATION")
    print("="*60)
    
    for i, email in enumerate(test_emails, 1):
        print(f"\n📧 Test Email {i}: {email['description']}")
        print(f"Sender: {email['sender_email']}")
        print(f"Preview: {email['email_body'][:100].strip()}...")
        
        # Classify the email
        result = classifier.classify_email(
            email_body=email['email_body'],
            sender_email=email['sender_email']
        )
        
        # Display results
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
        else:
            prediction = result['prediction']
            probability = result['probability']
            confidence = result['confidence']
            
            icon = "✅" if result['is_job_related'] else "❌"
            print(f"{icon} Prediction: {prediction}")
            print(f"📊 Probability: {probability:.1%}")
            print(f"🎯 Confidence: {confidence}")
    
    # Test batch processing
    print(f"\n" + "="*60)
    print("TESTING BATCH PROCESSING")
    print("="*60)
    
    batch_emails = [
        {"email_body": "Thank you for applying. We'll be in touch soon.", "sender_email": "hr@company.com"},
        {"email_body": "Your subscription will renew tomorrow.", "sender_email": "billing@service.com"},
        {"email_body": "Phone interview scheduled for Tuesday at 2 PM.", "sender_email": "recruiter@tech.com"},
        {"email_body": "Flash sale - 50% off everything!", "sender_email": "marketing@retailer.com"}
    ]
    
    batch_results = classifier.classify_batch(batch_emails)
    
    job_related_count = 0
    for i, result in enumerate(batch_results):
        email = batch_emails[i]
        if result.get('is_job_related'):
            job_related_count += 1
            status = "JOB RELATED ✅"
        else:
            status = "NOT JOB ❌"
        
        print(f"Email {i+1}: {status} (confidence: {result.get('confidence', 'unknown')})")
        print(f"  Content: {email['email_body'][:50]}...")
        print(f"  Sender: {email['sender_email']}")
        print()
    
    print(f"📊 Batch Summary:")
    print(f"   Total emails processed: {len(batch_results)}")
    print(f"   Job-related emails: {job_related_count}")
    print(f"   Non-job emails: {len(batch_results) - job_related_count}")
    
    # Integration tips
    print(f"\n" + "="*60)
    print("INTEGRATION TIPS")
    print("="*60)
    print("""
1. Model Loading:
   - Load the model once at application startup
   - Store the classifier instance for reuse
   - Check health periodically

2. Error Handling:
   - Always check for 'error' key in results
   - Handle network/file system issues gracefully
   - Log errors for debugging

3. Performance:
   - Use batch processing for multiple emails
   - Consider async processing for large volumes
   - Monitor processing time and accuracy

4. Monitoring:
   - Track confidence score distributions
   - Monitor for accuracy degradation
   - Collect user feedback for improvement

5. Production Deployment:
   - Ensure model files are accessible
   - Set up health check endpoints
   - Implement proper logging
   - Consider model versioning
    """)
    
    print("\n🎉 Integration example completed successfully!")
    print("You can now integrate this classifier into your application.")

if __name__ == "__main__":
    main()