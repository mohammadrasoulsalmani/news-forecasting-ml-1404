"""
مدل‌های پیش‌بینی تعامل با اخبار.
شامل:
- NewsForecaster: نسخه ساده‌شده LSTM (همان کد موجود)
- LSTMForecasterSF / MFN: مدل‌های مقاله (اختیاری)
"""

import torch
import torch.nn as nn
from config.settings import Config

class NewsForecaster(nn.Module):
    """
    LSTM دوجهته ساده برای پیش‌بینی بردار 7‌تایی تعاملات.
    خروجی embedding (برای خوشه‌بندی) و پیش‌بینی.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.NUM_STANCES,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL
        )
        lstm_out_dim = config.HIDDEN_DIM * (2 if config.BIDIRECTIONAL else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.NUM_STANCES)
        )
    
    def forward(self, x, return_embedding=True):
        """
        x: (batch, seq_len, 7)
        return_embedding: اگر True باشد، embedding (batch, lstm_out_dim) نیز برگردانده می‌شود.
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # آخرین hidden state از هر دو جهت
        if self.config.BIDIRECTIONAL:
            h_forward = h_n[-2, :, :]   # آخرین لایه، جهت forward
            h_backward = h_n[-1, :, :]  # آخرین لایه، جهت backward
            embedding = torch.cat([h_forward, h_backward], dim=1)
        else:
            embedding = h_n[-1, :, :]
        output = self.fc(embedding)
        if return_embedding:
            return embedding, output
        return output

