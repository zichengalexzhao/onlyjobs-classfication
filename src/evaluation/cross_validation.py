#!/usr/bin/env python3
"""
Rapid Cross-Validation Analysis
Quick CV analysis with 5-fold and optimized parameters.
"""

import json
import os
import pickle
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load training data quickly."""
    print("📂 Loading data...")
    
    data_path = "../data/ml_models/processed_features.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Combine train + val
    X = np.vstack([data['train_data']['features'], data['val_data']['features']])
    y = np.hstack([data['train_data']['labels'], data['val_data']['labels']])
    
    print(f"✅ Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def rapid_cv():
    """Rapid 5-fold CV on key models."""
    print("\n⚡ Rapid 5-Fold Cross-Validation")
    print("="*50)
    
    X, y = load_data()
    
    # Optimized models for speed
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=2),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=50),
        'SVM': SVC(random_state=42, C=1.0)
    }
    
    # 5-fold CV for speed
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 {name}...")
        
        try:
            # Just F1 and accuracy for speed
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=1)
            acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
            
            results[name] = {
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'accuracy_mean': np.mean(acc_scores),
                'accuracy_std': np.std(acc_scores),
                'f1_scores': f1_scores.tolist()
            }
            
            print(f"   F1: {results[name]['f1_mean']:.4f} ± {results[name]['f1_std']:.4f}")
            print(f"   Acc: {results[name]['accuracy_mean']:.4f} ± {results[name]['accuracy_std']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Rankings
    print(f"\n🏆 RANKINGS:")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_mean'], reverse=True)
    
    for i, (name, metrics) in enumerate(sorted_models, 1):
        print(f"   {i}. {name}: F1={metrics['f1_mean']:.4f}, Acc={metrics['accuracy_mean']:.4f}")
    
    # Best model
    best_name, best_metrics = sorted_models[0]
    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   F1-Score: {best_metrics['f1_mean']:.4f} ± {best_metrics['f1_std']:.4f}")
    
    # Save quick results
    os.makedirs("../data/ml_models", exist_ok=True)
    with open("../data/ml_models/rapid_cv_results.json", 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'best_model': best_name,
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Rapid CV complete! Best: {best_name}")
    return best_name, results

if __name__ == "__main__":
    rapid_cv()