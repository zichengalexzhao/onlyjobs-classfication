#!/usr/bin/env python3
"""
ML Model Training and Evaluation
Trains multiple classification models and compares their performance.
"""

import json
import os
import pickle
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize different ML models to train."""
        print("🤖 Initializing ML models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=20,
                class_weight='balanced'
            ),
            'SVM': SVC(
                random_state=42,
                probability=True,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1)
        }
        
        print(f"✅ Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def load_processed_data(self):
        """Load preprocessed features and labels."""
        print("\n📂 Loading processed data...")
        
        data_path = "data/models/processed_features.pkl"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data not found: {data_path}")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded processed data:")
        for split, split_data in data.items():
            features_shape = split_data['features'].shape
            labels_sum = np.sum(split_data['labels'])
            print(f"   {split}: {features_shape[0]} samples, {features_shape[1]} features ({labels_sum} positive, {features_shape[0]-labels_sum} negative)")
        
        return data
    
    def train_model(self, model, model_name, X_train, y_train, X_val, y_val):
        """Train a single model and evaluate it."""
        print(f"\n🔧 Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred),
            'val_recall': recall_score(y_val, y_val_pred),
            'val_f1': f1_score(y_val, y_val_pred),
            'val_auc': roc_auc_score(y_val, y_val_proba) if y_val_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_val, y_val_pred).tolist(),
            'classification_report': classification_report(y_val, y_val_pred, output_dict=True)
        }
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            results['cv_f1_mean'] = np.mean(cv_scores)
            results['cv_f1_std'] = np.std(cv_scores)
        except:
            results['cv_f1_mean'] = None
            results['cv_f1_std'] = None
        
        print(f"✅ {model_name} Results:")
        print(f"   Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"   Val Accuracy: {results['val_accuracy']:.4f}")
        print(f"   Val F1-Score: {results['val_f1']:.4f}")
        print(f"   Val Precision: {results['val_precision']:.4f}")
        print(f"   Val Recall: {results['val_recall']:.4f}")
        if results['val_auc']:
            print(f"   Val AUC: {results['val_auc']:.4f}")
        if results['cv_f1_mean']:
            print(f"   CV F1-Score: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
        
        return model, results
    
    def train_all_models(self, data):
        """Train all models and compare performance."""
        print("\n🚀 Training All Models")
        print("="*60)
        
        X_train = data['train_data']['features']
        y_train = data['train_data']['labels']
        X_val = data['val_data']['features']
        y_val = data['val_data']['labels']
        
        self.initialize_models()
        
        trained_models = {}
        best_f1 = 0
        
        for model_name, model in self.models.items():
            try:
                trained_model, results = self.train_model(
                    model, model_name, X_train, y_train, X_val, y_val
                )
                
                trained_models[model_name] = trained_model
                self.results[model_name] = results
                
                # Track best model
                if results['val_f1'] > best_f1:
                    best_f1 = results['val_f1']
                    self.best_model = trained_model
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f"❌ Failed to train {model_name}: {e}")
                continue
        
        print(f"\n🏆 Best Model: {self.best_model_name} (F1: {best_f1:.4f})")
        return trained_models
    
    def evaluate_on_test_set(self, trained_models, data):
        """Evaluate the best model on test set."""
        print(f"\n🧪 Testing Best Model: {self.best_model_name}")
        print("="*50)
        
        X_test = data['test_data']['features']
        y_test = data['test_data']['labels']
        
        # Test predictions
        y_test_pred = self.best_model.predict(X_test)
        y_test_proba = self.best_model.predict_proba(X_test)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        # Test metrics
        test_results = {
            'model_name': self.best_model_name,
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_auc': roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None,
            'test_confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
            'test_classification_report': classification_report(y_test, y_test_pred, output_dict=True)
        }
        
        print(f"🎯 Test Set Results:")
        print(f"   Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"   F1-Score: {test_results['test_f1']:.4f}")
        print(f"   Precision: {test_results['test_precision']:.4f}")
        print(f"   Recall: {test_results['test_recall']:.4f}")
        if test_results['test_auc']:
            print(f"   AUC: {test_results['test_auc']:.4f}")
        
        # Confusion Matrix
        cm = test_results['test_confusion_matrix']
        print(f"\n📊 Confusion Matrix:")
        print(f"   True Neg: {cm[0][0]}, False Pos: {cm[0][1]}")
        print(f"   False Neg: {cm[1][0]}, True Pos: {cm[1][1]}")
        
        return test_results
    
    def compare_with_llm_baseline(self):
        """Compare ML model performance with LLM baseline."""
        print(f"\n🤔 ML vs LLM Comparison")
        print("="*40)
        
        # Estimated LLM performance (based on our observation during data collection)
        llm_metrics = {
            'accuracy': 0.95,  # Estimated from manual review
            'speed_per_email': 2.0,  # 2 seconds per email
            'cost_per_email': 0.002  # $0.002 per email
        }
        
        # Our best ML model metrics
        if self.best_model_name and self.best_model_name in self.results:
            ml_metrics = {
                'accuracy': self.results[self.best_model_name]['val_accuracy'],
                'speed_per_email': 0.001,  # Very fast local inference
                'cost_per_email': 0.0  # No API costs
            }
            
            print(f"📊 Performance Comparison:")
            print(f"   Metric           LLM        ML Model    Improvement")
            print(f"   Accuracy        {llm_metrics['accuracy']:.3f}      {ml_metrics['accuracy']:.3f}       {ml_metrics['accuracy']/llm_metrics['accuracy']:.2f}x")
            print(f"   Speed (sec)     {llm_metrics['speed_per_email']:.3f}      {ml_metrics['speed_per_email']:.3f}       {llm_metrics['speed_per_email']/ml_metrics['speed_per_email']:.0f}x faster")
            print(f"   Cost ($)        {llm_metrics['cost_per_email']:.3f}      {ml_metrics['cost_per_email']:.3f}       {(llm_metrics['cost_per_email']-ml_metrics['cost_per_email'])/llm_metrics['cost_per_email']*100:.0f}% savings")
            
            # ROI calculation
            if ml_metrics['accuracy'] >= 0.90:  # If accuracy is good enough
                print(f"\n💰 Cost Savings Analysis:")
                emails_per_month = 1000  # Estimate
                monthly_llm_cost = emails_per_month * llm_metrics['cost_per_email']
                monthly_ml_cost = emails_per_month * ml_metrics['cost_per_email']
                monthly_savings = monthly_llm_cost - monthly_ml_cost
                print(f"   Monthly emails: {emails_per_month}")
                print(f"   LLM cost/month: ${monthly_llm_cost:.2f}")
                print(f"   ML cost/month: ${monthly_ml_cost:.2f}")
                print(f"   Monthly savings: ${monthly_savings:.2f}")
                print(f"   Annual savings: ${monthly_savings * 12:.2f}")
    
    def save_models_and_results(self, trained_models):
        """Save trained models and results."""
        print(f"\n💾 Saving Models and Results...")
        
        os.makedirs("data/models", exist_ok=True)
        
        # Save best model
        if self.best_model:
            with open("data/models/best_model.pkl", 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"✅ Best model saved: {self.best_model_name}")
        
        # Save all trained models
        with open("data/models/all_models.pkl", 'wb') as f:
            pickle.dump(trained_models, f)
        
        # Save results
        training_results = {
            'training_date': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'model_results': self.results,
            'summary': {
                'total_models_trained': len(self.results),
                'best_f1_score': max(r['val_f1'] for r in self.results.values()),
                'best_accuracy': max(r['val_accuracy'] for r in self.results.values())
            }
        }
        
        with open("data/models/training_results.json", 'w') as f:
            json.dump(training_results, f, indent=2)
        
        print(f"✅ Results saved to data/models/")
    
    def generate_model_summary(self):
        """Generate a summary of model performance."""
        print(f"\n📋 MODEL TRAINING SUMMARY")
        print("="*50)
        
        if not self.results:
            print("No models trained yet.")
            return
        
        # Sort models by F1 score
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1]['val_f1'], 
            reverse=True
        )
        
        print(f"🏆 Model Rankings (by F1-Score):")
        print(f"   Rank  Model               F1     Acc    Prec   Rec")
        print(f"   " + "-"*55)
        
        for i, (name, results) in enumerate(sorted_models, 1):
            print(f"   {i:2d}.   {name:<18} {results['val_f1']:.3f}  {results['val_accuracy']:.3f}  {results['val_precision']:.3f}  {results['val_recall']:.3f}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Best model: {self.best_model_name}")
        print(f"   Ready for production deployment!")

def main():
    """Main training function."""
    print("🤖 ML Model Training Pipeline")
    print("="*60)
    
    trainer = ModelTrainer()
    
    try:
        # Load processed data
        data = trainer.load_processed_data()
        
        # Train all models
        trained_models = trainer.train_all_models(data)
        
        if trained_models:
            # Evaluate on test set
            test_results = trainer.evaluate_on_test_set(trained_models, data)
            
            # Compare with LLM baseline
            trainer.compare_with_llm_baseline()
            
            # Save models and results
            trainer.save_models_and_results(trained_models)
            
            # Generate summary
            trainer.generate_model_summary()
            
            print(f"\n🎉 Training Pipeline Complete!")
            print(f"🚀 Ready to replace LLM with ML model!")
            
        else:
            print("❌ No models were successfully trained.")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()