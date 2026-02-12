"""
پیش‌پردازش: تبدیل داده‌های خام به توالی‌های زمانی.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from config.settings import Config

class SequenceBuilder:
    """تبدیل تعاملات هر کاربر به توالی‌های 8 قدمی و برچسب‌ها."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def build(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        ورودی: DataFrame شامل ستون‌های user_id, timestamp, stances
        خروجی: sequences با shape (n_samples, seq_len, 7)
                labels   با shape (n_samples, 7)
        """
        df['quarter'] = df['timestamp'].dt.to_period('Q')
        user_sequences = []
        user_labels = []
        
        for user_id, user_df in tqdm(df.groupby('user_id'), desc="👥 Processing users"):
            user_df = user_df.sort_values('timestamp')
            
            # شمارش تعاملات در هر سه‌ماهه
            quarterly_counts = []
            for _, quarter_df in user_df.groupby('quarter'):
                counts = np.zeros(self.config.NUM_STANCES, dtype=np.float32)
                for stances in quarter_df['stances']:
                    for stance in stances:
                        idx = int(stance) + 3   # -3 → 0, ..., +3 → 6
                        if 0 <= idx < self.config.NUM_STANCES:
                            counts[idx] += 1
                quarterly_counts.append(counts)
            
            # ساخت توالی‌های پنجره‌ای
            seq_len = self.config.SEQ_LENGTH
            if len(quarterly_counts) >= seq_len + 1:
                for i in range(len(quarterly_counts) - seq_len):
                    seq = quarterly_counts[i:i+seq_len]
                    label = quarterly_counts[i+seq_len]
                    user_sequences.append(seq)
                    user_labels.append(label)
        
        sequences = np.array(user_sequences, dtype=np.float32)
        labels = np.array(user_labels, dtype=np.float32)
        print(f"✅ Created {len(sequences)} sequences.")
        return sequences, labels