#!/usr/bin/env python3
"""
Data Collection Script
Collects job-related and non-job-related emails for training data.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_collection.gmail_collector import CostEfficientCollector

def main():
    parser = argparse.ArgumentParser(
        description="Collect email data for job classification training"
    )
    parser.add_argument(
        "--positive-samples", 
        type=int, 
        default=1000,
        help="Number of job-related emails to collect"
    )
    parser.add_argument(
        "--negative-samples", 
        type=int, 
        default=1000,
        help="Number of non-job-related emails to collect"
    )
    parser.add_argument(
        "--budget", 
        type=float, 
        default=5.0,
        help="Maximum budget in USD for OpenAI API calls"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/raw",
        help="Output directory for collected data"
    )
    
    args = parser.parse_args()
    
    print("🚀 OnlyJobs Data Collection")
    print("="*50)
    print(f"Target: {args.positive_samples} positive + {args.negative_samples} negative")
    print(f"Budget: ${args.budget:.2f}")
    print(f"Output: {args.output_dir}")
    
    # Initialize collector
    collector = CostEfficientCollector(max_budget_usd=args.budget)
    
    try:
        # Collect data
        data = collector.collect_binary_classification_data(
            target_positive=args.positive_samples,
            target_negative=args.negative_samples
        )
        
        if data['positive_examples'] or data['negative_examples']:
            # Create training dataset
            collector.create_real_training_dataset(
                data['positive_examples'], 
                data['negative_examples']
            )
            
            print(f"\n✅ Data collection successful!")
            print(f"Collected: {len(data['positive_examples'])} positive, {len(data['negative_examples'])} negative")
            print(f"Cost: ${collector.estimated_cost:.2f}")
            
        else:
            print("\n❌ No data collected. Check your Gmail API setup.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Data collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()