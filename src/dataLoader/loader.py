"""
JSON Data Loader with Sampling Support.
Loads Twitter data and converts to DataFrame with progress tracking.
"""

import json
import pandas as pd
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from config.settings import Config

class DataLoader:
    """Load raw Twitter data from JSON file with random sampling."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load(self, 
             filepath: Optional[Path] = None, 
             sample_size: Optional[int] = None,
             sample_fraction: Optional[float] = None) -> pd.DataFrame:
        """
        Load JSON file and create DataFrame with sampling.
        
        Args:
            filepath: Path to JSON file
            sample_size: Number of records to sample (takes priority)
            sample_fraction: Fraction of records to sample (0.0-1.0)
        
        Returns:
            DataFrame with columns: user_id, timestamp, sources, stances
        """
        # Set paths and sampling parameters
        if filepath is None:
            filepath = self.config.DATA_PATH
        
        # Determine sampling method
        if sample_size is None and sample_fraction is None:
            # Default: use config values
            sample_size = self.config.SAMPLE_SIZE
            sample_fraction = self.config.SAMPLE_FRACTION
        
        print(f"📂 Loading data from: {filepath}")
        
        # Load full JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_records = len(data)
        print(f"📊 Total records in dataset: {total_records:,}")
        
        # Apply sampling
        all_keys = list(data.keys())
        
        if sample_size is not None:
            # Sample by count
            n_samples = min(sample_size, total_records)
            sampled_keys = random.sample(all_keys, n_samples)
            print(f"🎯 Sampling {n_samples:,} records ({n_samples/total_records:.1%})")
            
        elif sample_fraction is not None:
            # Sample by fraction
            n_samples = int(total_records * sample_fraction)
            sampled_keys = random.sample(all_keys, n_samples)
            print(f"🎯 Sampling {sample_fraction:.0%} of data → {n_samples:,} records")
        else:
            # No sampling
            sampled_keys = all_keys
            print(f"🎯 Using full dataset ({total_records:,} records)")
        
        # Load sampled records with progress bar
        records = []
        for key in tqdm(sampled_keys, desc="📥 Loading records", unit="rec"):
            value = data[key]
            records.append({
                'user_id': value['user_id_anonymized'],
                'timestamp': pd.to_datetime(value['created_at']),
                'sources': value['news sources'],
                'stances': value['partisan stance']
            })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Print statistics
        print(f"\n✅ Successfully loaded:")
        print(f"   • {len(df):,} total records")
        print(f"   • {df['user_id'].nunique():,} unique users")
        print(f"   • Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        print(f"   • Avg records per user: {df.groupby('user_id').size().mean():.1f}")
        
        return df
    
    def load_sample(self, fraction: float = 0.1) -> pd.DataFrame:
        """Convenience method to load a sample."""
        return self.load(sample_fraction=fraction)
    
    def load_full(self) -> pd.DataFrame:
        """Load entire dataset (use with caution)."""
        return self.load(sample_size=None, sample_fraction=None)