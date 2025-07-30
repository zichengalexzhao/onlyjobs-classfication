#!/usr/bin/env python3
"""
Analyze Manual Review Results
Reads the Excel file with manual review annotations and evaluates model performance.
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
from pathlib import Path

def load_evaluation_results(excel_file):
    """Load the Excel file with evaluation results"""
    print(f"📊 Loading evaluation results from: {excel_file}")
    
    try:
        df = pd.read_excel(excel_file, sheet_name='Email Evaluation')
        print(f"✅ Loaded {len(df)} email evaluations")
        return df
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None

def analyze_predictions(df):
    """Analyze ML predictions vs manual review"""
    print("\n🔍 Analyzing ML Model Performance")
    print("=" * 50)
    
    # Clean up the manual review column
    df['Manual Review Clean'] = df['Is Prediction Correct?'].fillna('').str.strip().str.lower()
    
    # Count total predictions
    total_emails = len(df)
    job_related_predictions = len(df[df['ML Prediction'] == 'JOB-RELATED'])
    non_job_predictions = len(df[df['ML Prediction'] == 'NON-JOB-RELATED'])
    
    print(f"📈 Overall Statistics:")
    print(f"   Total emails: {total_emails}")
    print(f"   Predicted JOB-RELATED: {job_related_predictions} ({job_related_predictions/total_emails*100:.1f}%)")
    print(f"   Predicted NON-JOB-RELATED: {non_job_predictions} ({non_job_predictions/total_emails*100:.1f}%)")
    
    # Analyze manual review results
    review_counts = Counter(df['Manual Review Clean'].values)
    print(f"\n📝 Manual Review Summary:")
    for review_type, count in review_counts.items():
        if review_type:  # Skip empty strings
            print(f"   {review_type.title()}: {count} ({count/total_emails*100:.1f}%)")
        else:
            print(f"   Not reviewed: {count} ({count/total_emails*100:.1f}%)")
    
    # Calculate performance metrics
    print(f"\n🎯 Performance Analysis:")
    
    # Count correct and incorrect predictions
    # NOTE: User only marked incorrect ones, so empty = correct
    incorrect_predictions = len(df[df['Manual Review Clean'] == 'incorrect'])
    unsure_predictions = len(df[df['Manual Review Clean'] == 'unsure'])
    explicit_correct = len(df[df['Manual Review Clean'] == 'correct'])
    not_marked = len(df[df['Manual Review Clean'] == ''])
    
    # Assume not marked = correct (user only marked incorrect ones)
    correct_predictions = explicit_correct + not_marked
    
    total_reviewed = total_emails  # All emails were reviewed
    accuracy = correct_predictions / total_reviewed * 100
    
    print(f"   Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_reviewed})")
    print(f"   Correct predictions: {correct_predictions} (including {not_marked} unmarked)")
    print(f"   Incorrect predictions: {incorrect_predictions}")
    print(f"   Unsure predictions: {unsure_predictions}")
    print(f"   Explicitly marked correct: {explicit_correct}")
    
    return {
        'total_emails': total_emails,
        'job_related_predictions': job_related_predictions,
        'non_job_predictions': non_job_predictions,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions,
        'unsure_predictions': unsure_predictions,
        'not_marked': not_marked,
        'accuracy': accuracy
    }

def analyze_incorrect_predictions(df):
    """Analyze the incorrect predictions in detail"""
    print(f"\n❌ Analyzing Incorrect Predictions")
    print("=" * 40)
    
    # Filter incorrect predictions
    incorrect_df = df[df['Is Prediction Correct?'].str.strip().str.lower() == 'incorrect'].copy()
    
    if len(incorrect_df) == 0:
        print("✅ No incorrect predictions found!")
        return
    
    print(f"Found {len(incorrect_df)} incorrect predictions:")
    print()
    
    for idx, row in incorrect_df.iterrows():
        print(f"📧 Email {idx + 1}:")
        print(f"   Subject: {row['Subject']}")
        print(f"   Sender: {row['Sender']}")
        print(f"   ML Prediction: {row['ML Prediction']}")
        print(f"   Confidence: {row['Confidence %']}")
        body_preview = str(row['Email Body'])[:100] if pd.notna(row['Email Body']) else "No body content"
        print(f"   Body Preview: {body_preview}...")
        print()
    
    # Analyze patterns in incorrect predictions
    print(f"🔍 Error Analysis:")
    
    # Analyze by prediction type
    incorrect_job_related = len(incorrect_df[incorrect_df['ML Prediction'] == 'JOB-RELATED'])
    incorrect_non_job = len(incorrect_df[incorrect_df['ML Prediction'] == 'NON-JOB-RELATED'])
    
    print(f"   False Positives (incorrectly predicted as JOB-RELATED): {incorrect_job_related}")
    print(f"   False Negatives (incorrectly predicted as NON-JOB-RELATED): {incorrect_non_job}")
    
    # Analyze confidence levels of incorrect predictions
    if len(incorrect_df) > 0:
        incorrect_df['Confidence Numeric'] = incorrect_df['Confidence %'].str.replace('%', '').astype(float)
        avg_confidence = incorrect_df['Confidence Numeric'].mean()
        print(f"   Average confidence of incorrect predictions: {avg_confidence:.1f}%")
        
        # Show confidence distribution
        high_conf_incorrect = len(incorrect_df[incorrect_df['Confidence Numeric'] >= 90])
        medium_conf_incorrect = len(incorrect_df[(incorrect_df['Confidence Numeric'] >= 70) & (incorrect_df['Confidence Numeric'] < 90)])
        low_conf_incorrect = len(incorrect_df[incorrect_df['Confidence Numeric'] < 70])
        
        print(f"   High confidence (≥90%) incorrect: {high_conf_incorrect}")
        print(f"   Medium confidence (70-89%) incorrect: {medium_conf_incorrect}")
        print(f"   Low confidence (<70%) incorrect: {low_conf_incorrect}")

def analyze_correct_predictions(df):
    """Analyze the correct predictions for patterns"""
    print(f"\n✅ Analyzing Correct Predictions")
    print("=" * 40)
    
    # Filter correct predictions
    correct_df = df[df['Is Prediction Correct?'].str.strip().str.lower() == 'correct'].copy()
    
    if len(correct_df) == 0:
        print("❌ No correct predictions found!")
        return
    
    print(f"Found {len(correct_df)} correct predictions")
    
    # Analyze by prediction type
    correct_job_related = len(correct_df[correct_df['ML Prediction'] == 'JOB-RELATED'])
    correct_non_job = len(correct_df[correct_df['ML Prediction'] == 'NON-JOB-RELATED'])
    
    print(f"   Correctly predicted JOB-RELATED: {correct_job_related}")
    print(f"   Correctly predicted NON-JOB-RELATED: {correct_non_job}")
    
    # Analyze confidence levels of correct predictions
    if len(correct_df) > 0:
        correct_df['Confidence Numeric'] = correct_df['Confidence %'].str.replace('%', '').astype(float)
        avg_confidence = correct_df['Confidence Numeric'].mean()
        print(f"   Average confidence of correct predictions: {avg_confidence:.1f}%")

def generate_recommendations(analysis_results, df):
    """Generate recommendations based on the analysis"""
    print(f"\n💡 Recommendations for Model Improvement")
    print("=" * 50)
    
    # Calculate error rates
    total_reviewed = analysis_results['total_emails']  # All emails were reviewed
    error_rate = analysis_results['incorrect_predictions'] / total_reviewed * 100
    
    print(f"📊 Current Performance:")
    print(f"   Error Rate: {error_rate:.1f}%")
    print(f"   Accuracy: {analysis_results['accuracy']:.1f}%")
    
    # Analyze prediction distribution
    job_rate = analysis_results['job_related_predictions'] / analysis_results['total_emails'] * 100
    
    print(f"\n🎯 Model Behavior Analysis:")
    print(f"   Job classification rate: {job_rate:.1f}% (very conservative)")
    
    if job_rate < 5:
        print("   ⚠️  Model is extremely conservative - may be missing many job-related emails")
    elif job_rate < 10:
        print("   ⚠️  Model is quite conservative - check for false negatives")
    
    # Specific recommendations
    print(f"\n🔧 Specific Recommendations:")
    
    incorrect_df = df[df['Is Prediction Correct?'].str.strip().str.lower() == 'incorrect']
    
    if len(incorrect_df) > 0:
        false_negatives = len(incorrect_df[incorrect_df['ML Prediction'] == 'NON-JOB-RELATED'])
        false_positives = len(incorrect_df[incorrect_df['ML Prediction'] == 'JOB-RELATED'])
        
        if false_negatives > false_positives:
            print("   1. Model is missing job-related emails (high false negative rate)")
            print("      → Consider lowering classification threshold")
            print("      → Add more diverse job-related training examples")
            print("      → Review feature engineering for job-related patterns")
        
        if false_positives > 0:
            print("   2. Model has some false positives")
            print("      → Review non-job emails that were misclassified")
            print("      → Add negative examples to training data")
        
        print("   3. Consider retraining with the manually reviewed emails as additional training data")
        print("   4. Implement confidence-based filtering for edge cases")
    
    print(f"\n📈 Next Steps:")
    print("   1. Complete manual review of all 100 emails")
    print("   2. Use this feedback to retrain the model")
    print("   3. Run another evaluation round to measure improvement")
    print("   4. Consider A/B testing different confidence thresholds")

def main():
    """Main analysis function"""
    print("🚀 Manual Review Analysis")
    print("=" * 60)
    
    # Find the most recent evaluation file
    import glob
    excel_files = glob.glob("model_evaluation_*.xlsx")
    
    if not excel_files:
        print("❌ No evaluation Excel files found!")
        print("   Please run the Gmail evaluation script first.")
        return
    
    # Use the most recent file
    excel_file = max(excel_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"📄 Using file: {excel_file}")
    
    # Load and analyze data
    df = load_evaluation_results(excel_file)
    if df is None:
        return
    
    # Perform analysis
    analysis_results = analyze_predictions(df)
    analyze_incorrect_predictions(df)
    analyze_correct_predictions(df)
    generate_recommendations(analysis_results, df)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Summary: {analysis_results['accuracy']:.1f}% accuracy on reviewed emails")

if __name__ == "__main__":
    main()