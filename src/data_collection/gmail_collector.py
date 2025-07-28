#!/usr/bin/env python3
"""
Cost-Efficient Real Email Data Collector
Focused on binary classification only - no expensive detail extraction.
Implements smart sampling and cost controls to minimize API usage.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "job-app-tracker"))

from scripts.gmail_fetch import fetch_emails, get_email_content, get_email_snippet
from scripts.process_emails import is_job_application

class CostEfficientCollector:
    def __init__(self, max_budget_usd=5.0):
        """
        Initialize collector with cost controls.
        
        Args:
            max_budget_usd: Maximum budget for OpenAI API calls (default: $5)
        """
        self.max_budget_usd = max_budget_usd
        self.api_calls_made = 0
        self.estimated_cost = 0.0
        self.cost_per_call = 0.002  # ~$0.002 per GPT-3.5-turbo call
        
        print(f"💰 Cost Control: Max budget ${max_budget_usd:.2f}")
        print(f"📊 Estimated calls available: {int(max_budget_usd / self.cost_per_call)}")
    
    def check_budget(self):
        """Check if we're within budget."""
        return self.estimated_cost < self.max_budget_usd
    
    def update_cost(self):
        """Update cost tracking."""
        self.api_calls_made += 1
        self.estimated_cost = self.api_calls_made * self.cost_per_call
        
        if self.api_calls_made % 100 == 0:
            print(f"💰 Cost update: {self.api_calls_made} calls, ~${self.estimated_cost:.2f}")
    
    def smart_sample_emails(self, target_samples=2000, max_fetch=5000):
        """
        Smart sampling strategy to get diverse emails efficiently.
        
        Args:
            target_samples: Number of samples we want
            max_fetch: Maximum emails to fetch from Gmail
        """
        
        print(f"\n📧 Smart Sampling Strategy:")
        print(f"  Target samples: {target_samples}")
        print(f"  Max fetch limit: {max_fetch}")
        
        # Fetch emails with limit
        try:
            messages = fetch_emails(since_hours=None)  # All emails
            print(f"  Available emails: {len(messages)}")
            
            # If we have more emails than max_fetch, sample randomly
            if len(messages) > max_fetch:
                print(f"  Sampling {max_fetch} emails from {len(messages)} available")
                random.seed(42)
                messages = random.sample(messages, max_fetch)
            
            return messages
            
        except Exception as e:
            print(f"❌ Error fetching emails: {e}")
            return []
    
    def collect_binary_classification_data(self, target_positive=1000, target_negative=1000):
        """
        Collect data focused ONLY on binary classification (job vs non-job).
        No expensive detail extraction.
        """
        
        print(f"\n🎯 Binary Classification Data Collection")
        print(f"Target: {target_positive} positive + {target_negative} negative examples")
        print("="*60)
        
        # Smart sampling
        max_fetch = (target_positive + target_negative) * 3  # 3x buffer for filtering
        messages = self.smart_sample_emails(
            target_samples=target_positive + target_negative,
            max_fetch=max_fetch
        )
        
        if not messages:
            return {"positive_examples": [], "negative_examples": []}
        
        positive_examples = []
        negative_examples = []
        processed_count = 0
        error_count = 0
        
        print(f"\n🔍 Processing {len(messages)} emails for binary classification...")
        
        for i, msg in enumerate(messages):
            # Budget check
            if not self.check_budget():
                print(f"\n💰 Budget limit reached! Stopping collection.")
                print(f"   Processed: {processed_count} emails")
                print(f"   Cost: ~${self.estimated_cost:.2f}")
                break
            
            # Target check
            if (len(positive_examples) >= target_positive and 
                len(negative_examples) >= target_negative):
                print(f"\n✅ Targets reached! Stopping collection.")
                break
            
            msg_id = msg['id']
            
            try:
                # Get email snippet (free, no API call)
                snippet = get_email_snippet(msg_id)
                
                # Get full email content (free, no API call)
                email_data = get_email_content(msg_id)
                full_content = email_data["content"]
                email_date = email_data["date"]
                
                # ONLY make OpenAI API call for binary classification
                # Skip if we already have enough of this type
                make_api_call = False
                
                if len(positive_examples) < target_positive or len(negative_examples) < target_negative:
                    make_api_call = True
                
                if make_api_call:
                    # Single API call for binary classification
                    is_job_related = is_job_application(snippet)
                    self.update_cost()
                    
                    # Create training example
                    training_example = {
                        "email_id": msg_id,
                        "snippet": snippet,
                        "full_content": full_content,
                        "date": email_date,
                        "label": "job_related" if is_job_related else "non_job_related",
                        "collected_at": datetime.now().isoformat(),
                        "source": "real_gmail_api"
                    }
                    
                    # Add to appropriate category
                    if is_job_related and len(positive_examples) < target_positive:
                        positive_examples.append(training_example)
                        print(f"✅ Job-related #{len(positive_examples)}: {snippet[:60]}...")
                    elif not is_job_related and len(negative_examples) < target_negative:
                        negative_examples.append(training_example)
                        print(f"❌ Non-job #{len(negative_examples)}: {snippet[:60]}...")
                
                processed_count += 1
                
                # Progress update
                if processed_count % 50 == 0:
                    print(f"\n📊 Progress:")
                    print(f"   Processed: {processed_count}/{len(messages)} emails")
                    print(f"   Positive: {len(positive_examples)}/{target_positive}")
                    print(f"   Negative: {len(negative_examples)}/{target_negative}")
                    print(f"   API calls: {self.api_calls_made} (~${self.estimated_cost:.2f})")
                    print(f"   Errors: {error_count}")
                
            except Exception as e:
                error_count += 1
                if error_count % 10 == 0:
                    print(f"⚠️ Errors: {error_count}")
                continue
        
        print(f"\n🎉 Collection Complete!")
        print(f"📊 Final Stats:")
        print(f"   Total processed: {processed_count} emails")
        print(f"   Positive examples: {len(positive_examples)}")
        print(f"   Negative examples: {len(negative_examples)}")
        print(f"   API calls made: {self.api_calls_made}")
        print(f"   Estimated cost: ${self.estimated_cost:.2f}")
        print(f"   Errors: {error_count}")
        
        return {
            "positive_examples": positive_examples,
            "negative_examples": negative_examples,
            "collection_stats": {
                "total_processed": processed_count,
                "api_calls_made": self.api_calls_made,
                "estimated_cost_usd": self.estimated_cost,
                "error_count": error_count,
                "collection_date": datetime.now().isoformat()
            }
        }
    
    def create_real_training_dataset(self, positive_examples, negative_examples):
        """Create training dataset from real email data."""
        
        print(f"\n⚖️ Creating Real Email Training Dataset...")
        
        # Balance the dataset
        min_size = min(len(positive_examples), len(negative_examples))
        balanced_positive = positive_examples[:min_size]
        balanced_negative = negative_examples[:min_size]
        
        print(f"Balanced dataset: {min_size} positive + {min_size} negative = {min_size * 2} total")
        
        # Combine and shuffle
        all_examples = balanced_positive + balanced_negative
        random.seed(42)
        random.shuffle(all_examples)
        
        # Create splits
        total = len(all_examples)
        train_size = int(0.7 * total)
        val_size = int(0.2 * total)
        
        train_data = all_examples[:train_size]
        val_data = all_examples[train_size:train_size + val_size]
        test_data = all_examples[train_size + val_size:]
        
        print(f"\nDataset splits:")
        print(f"  Training: {len(train_data)} examples")
        print(f"  Validation: {len(val_data)} examples")
        print(f"  Test: {len(test_data)} examples")
        
        # Save datasets
        os.makedirs("../data/real_email_training", exist_ok=True)
        
        datasets = {
            "train_data.json": train_data,
            "val_data.json": val_data,
            "test_data.json": test_data,
            "positive_examples.json": balanced_positive,
            "negative_examples.json": balanced_negative
        }
        
        for filename, data in datasets.items():
            filepath = f"../data/real_email_training/{filename}"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved {len(data)} examples to {filepath}")
        
        # Save statistics
        stats = {
            "creation_date": datetime.now().isoformat(),
            "total_examples": len(all_examples),
            "positive_examples": len(balanced_positive),
            "negative_examples": len(balanced_negative),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "data_source": "real_gmail_api",
            "api_cost_usd": self.estimated_cost,
            "api_calls_made": self.api_calls_made
        }
        
        with open("../data/real_email_training/dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Saved dataset statistics")
        
        # Show sample examples
        print(f"\n📋 Sample real email examples:")
        for i, example in enumerate(train_data[:3]):
            print(f"\nExample {i+1} (Label: {example['label']}):")
            snippet = example['snippet'][:100] + "..." if len(example['snippet']) > 100 else example['snippet']
            print(f"  Snippet: {snippet}")
            print(f"  Date: {example['date']}")
            print(f"  Content length: {len(example['full_content'])} chars")
        
        return train_data, val_data, test_data

def main():
    """Main cost-efficient collection function."""
    
    print("💰 Cost-Efficient Real Email Data Collection")
    print("Focus: Binary classification only (job vs non-job)")
    print("Cost Control: ~$5 budget limit")
    print("="*70)
    
    # Initialize collector with budget
    collector = CostEfficientCollector(max_budget_usd=5.0)
    
    # Collect real email data
    print("Phase 1: Collecting real email data...")
    data = collector.collect_binary_classification_data(
        target_positive=1000,  # Reasonable target
        target_negative=1000   # Balanced
    )
    
    positive_examples = data["positive_examples"]
    negative_examples = data["negative_examples"]
    
    if len(positive_examples) == 0 and len(negative_examples) == 0:
        print("❌ No data collected. Check Gmail API setup.")
        return
    
    # Create training dataset
    print("\nPhase 2: Creating training dataset...")
    train_data, val_data, test_data = collector.create_real_training_dataset(
        positive_examples, negative_examples
    )
    
    print(f"\n🎉 Real Email Data Collection Complete!")
    print(f"\n📊 Summary:")
    print(f"  Real email examples: {len(train_data) + len(val_data) + len(test_data)}")
    print(f"  API calls made: {collector.api_calls_made}")
    print(f"  Estimated cost: ${collector.estimated_cost:.2f}")
    print(f"  Data saved to: ../data/real_email_training/")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Compare real vs synthetic data quality")
    print(f"2. Train ML models on real email data")
    print(f"3. Evaluate binary classification performance")

if __name__ == "__main__":
    main()