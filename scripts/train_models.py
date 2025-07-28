#!/usr/bin/env python3
"""
Model Training Script
Trains and evaluates ML models for email classification.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.classifier import ModelTrainer
from feature_engineering.feature_pipeline import process_dataset_features

def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for email classification"
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        choices=["all", "random_forest", "gradient_boosting", "svm", "logistic"],
        default="all",
        help="Algorithm to train (default: all)"
    )
    parser.add_argument(
        "--cv-folds", 
        type=int, 
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/raw",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--tune-hyperparameters", 
        action="store_true",
        help="Perform hyperparameter tuning"
    )
    
    args = parser.parse_args()
    
    print("🤖 OnlyJobs Model Training")
    print("="*50)
    print(f"Algorithm: {args.algorithm}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Process features if not already done
        print("\n🔧 Processing features...")
        process_dataset_features()
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Load data
        data = trainer.load_processed_data()
        
        if args.algorithm == "all":
            # Train all models
            trained_models = trainer.train_all_models(data)
        else:
            # Train specific model
            # This would need implementation for single model training
            print(f"Training single model: {args.algorithm}")
            trained_models = trainer.train_all_models(data)  # For now, train all
        
        if trained_models:
            # Evaluate on test set
            test_results = trainer.evaluate_on_test_set(trained_models, data)
            
            # Save models and results
            trainer.save_models_and_results(trained_models)
            
            # Generate summary
            trainer.generate_model_summary()
            
            print(f"\n🎉 Training complete!")
            print(f"Best model: {trainer.best_model_name}")
            print(f"Models saved to: {args.output_dir}")
            
        else:
            print("\n❌ No models were successfully trained.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()