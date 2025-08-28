#!/usr/bin/env python3
"""
Fix Feature Pipeline Implementation
Properly saves the complete GeneralizedEmailFeatureExtractor object instead of just components.
"""

import sys
import os
sys.path.insert(0, 'src')

import pickle
import pandas as pd
from feature_engineering.generalized_feature_pipeline import GeneralizedEmailFeatureExtractor

def fix_feature_pipeline():
    """Fix the feature pipeline by saving the complete extractor object."""
    
    print("=== Fixing Feature Pipeline Implementation ===")
    
    # File paths
    data_dir = "data/enhanced_training"
    models_dir = "data/models"
    train_data_path = f"{data_dir}/train_data.csv"
    pipeline_path = f"{models_dir}/generalized_feature_pipeline.pkl"
    
    # Check if training data exists
    if not os.path.exists(train_data_path):
        print(f"ERROR: Training data not found at {train_data_path}")
        return False
    
    print(f"Loading training data from: {train_data_path}")
    
    # Load training data
    train_df = pd.read_csv(train_data_path)
    print(f"Loaded {len(train_df)} training emails")
    
    # Create a new GeneralizedEmailFeatureExtractor
    print("Creating new GeneralizedEmailFeatureExtractor...")
    extractor = GeneralizedEmailFeatureExtractor()
    
    # Fit the extractor on training data
    print("Fitting feature extractor on training data...")
    try:
        features, feature_names = extractor.extract_all_features(train_df, max_tfidf_features=500, fit=True)
        print(f"Successfully extracted {features.shape[1]} features from {features.shape[0]} emails")
    except Exception as e:
        print(f"ERROR: Failed to fit feature extractor: {e}")
        return False
    
    # Save the COMPLETE extractor object (not just components)
    print(f"Saving complete GeneralizedEmailFeatureExtractor object to: {pipeline_path}")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the complete extractor object using pickle directly
    try:
        with open(pipeline_path, 'wb') as f:
            pickle.dump(extractor, f)
        print("✓ Successfully saved the complete GeneralizedEmailFeatureExtractor object!")
    except Exception as e:
        print(f"ERROR: Failed to save extractor: {e}")
        return False
    
    # Verify the saved object
    print("Verifying saved pipeline...")
    try:
        with open(pipeline_path, 'rb') as f:
            loaded_extractor = pickle.load(f)
        
        if isinstance(loaded_extractor, GeneralizedEmailFeatureExtractor):
            print("✓ Verification successful: Saved object is GeneralizedEmailFeatureExtractor")
            print(f"✓ Extractor is fitted: {loaded_extractor.fitted}")
            print(f"✓ Has extract_all_features method: {hasattr(loaded_extractor, 'extract_all_features')}")
            
            # Quick test with sample data
            print("Testing with sample email...")
            test_data = pd.DataFrame([{
                'email_body': 'Thank you for applying to our Software Engineer position.',
                'sender': 'hr@company.com',
                'sender_email': 'hr@company.com',
                'date': 'Mon, 26 Aug 2024 10:00:00 +0000',
                'parsed_datetime': '2024-08-26T10:00:00',
                'date_only': '2024-08-26',
                'week': 35,
                'month': 8,
                'year': 2024,
                'days_since': 2,
                'is_talent_acquisition': False
            }])
            
            test_features, _ = loaded_extractor.extract_all_features(test_data, fit=False)
            print(f"✓ Test successful: Extracted {test_features.shape[1]} features from test email")
            
        else:
            print(f"ERROR: Saved object is {type(loaded_extractor)}, not GeneralizedEmailFeatureExtractor")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to verify saved pipeline: {e}")
        return False
    
    print("\n=== Fix Complete ===")
    print("The feature pipeline now contains the complete GeneralizedEmailFeatureExtractor object")
    print("Production code can now use extractor.extract_all_features() directly")
    
    return True

if __name__ == "__main__":
    success = fix_feature_pipeline()
    if success:
        print("\n✓ Feature pipeline fix completed successfully!")
        exit(0)
    else:
        print("\n✗ Feature pipeline fix failed!")
        exit(1)