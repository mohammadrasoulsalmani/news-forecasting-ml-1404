"""
models.py
ماژول مدل‌های پیش‌بینی تعامل با اخبار در توییتر
شامل: LSTM تک‌ویژگی، LSTM چندویژگی و BERT
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoConfig


class BaseForecaster(nn.Module, ABC):
    """کلاس پایه برای همه مدل‌های پیش‌بینی"""
    
    def __init__(self, output_size: int = 7):
        super(BaseForecaster, self).__init__()
        self.output_size = output_size
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def get_embedding_dim(self) -> int:
        """بُعد بردار embedding برای clustering"""
        if hasattr(self, 'embedding_dim'):
            return self.embedding_dim
        return 128


class LSTMForecasterSF(BaseForecaster):
    """
    Single Feature Network (SFN)
    پیش‌بینی با یک نوع ویژگی (تعداد تعاملات یا متن)
    
    Args:
        input_size: تعداد ویژگی‌های ورودی (7 برای count، 3072 برای text)
        hidden_dim_lstm: تعداد واحدهای LSTM
        num_layers: تعداد لایه‌های LSTM
        bidirectional: دوطرفه بودن LSTM
        output_size: 7 (تعداد دسته‌بندی‌های سیاسی)
        dropout: مقدار Dropout
        activation: تابع فعال‌سازی
    """
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_dim_lstm: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        output_size: int = 7,
        dropout: float = 0.2,
        activation: str = "sigmoid"
    ):
        super(LSTMForecasterSF, self).__init__(output_size)
        
        self.input_size = input_size
        self.hidden_dim_lstm = hidden_dim_lstm
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding_dim = hidden_dim_lstm * self.num_directions
        
        # لایه ورودی برای پروجکشن ویژگی‌ها (اختیاری)
        if input_size > 100:  # برای text features
            self.input_proj = nn.Linear(input_size, hidden_dim_lstm)
            lstm_input_size = hidden_dim_lstm
        else:
            self.input_proj = None
            lstm_input_size = input_size
        
        # لایه LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim_lstm,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # تابع فعال‌سازی
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:  # sigmoid
            self.activation = nn.Sigmoid()
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # لایه خروجی
        self.fc_out = nn.Linear(self.embedding_dim, output_size)
        
        # تنظیم وزن‌ها
        self._init_weights()
    
    def _init_weights(self):
        """مقداردهی اولیه وزن‌ها"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(
        self, 
        feats: torch.Tensor,
        return_embedding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feats: ورودی [batch_size, seq_len, input_size]
            return_embedding: برگرداندن embedding برای clustering
        
        Returns:
            embedding: [batch_size, embedding_dim]
            output: [batch_size, output_size]
        """
        # پروجکشن ویژگی‌ها در صورت نیاز
        if self.input_proj is not None:
            feats = self.activation(self.input_proj(feats))
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(feats)
        
        # استخراج آخرین hidden state
        batch_size = feats.shape[0]
        
        if self.bidirectional:
            h_forward = h_n[-2, :, :]  # آخرین لایه، جهت forward
            h_backward = h_n[-1, :, :]  # آخرین لایه، جهت backward
            embedding = torch.cat([h_forward, h_backward], dim=-1)
        else:
            embedding = h_n[-1, :, :]
        
        embedding = self.dropout(embedding)
        
        # خروجی نهایی
        output = self.fc_out(embedding)
        
        if return_embedding:
            return embedding, output
        return embedding, output


class LSTMForecasterMF(BaseForecaster):
    """
    Multiple Feature Network (MFN)
    پیش‌بینی با ترکیب چند ویژگی:
    - Count features (7 بعد)
    - Text features (3072 بعد → پروجکشن)
    - Time features (4 بعد)
    - Hashtag features (1536 بعد)
    - Mention features (اختیاری)
    """
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_dim_lstm: int = 256,
        hidden_text: int = 128,
        hidden_inter: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        output_size: int = 7,
        activation: str = "sigmoid"
    ):
        super(LSTMForecasterMF, self).__init__(output_size)
        
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding_dim = hidden_inter
        
        # 1. پروجکشن ویژگی‌های متن (3072 → hidden_text)
        self.fc_text = nn.Sequential(
            nn.Linear(3072, hidden_text),
            nn.LayerNorm(hidden_text),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. پروجکشن ویژگی‌های هشتگ (1536 → 64)
        self.fc_hashtag = nn.Sequential(
            nn.Linear(1536, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. LSTM روی ویژگی‌های ترکیبی
        lstm_input_size = input_size + hidden_text + 4  # count + text_proj + time
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim_lstm,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 4. لایه میانی (intermediate)
        lstm_output_dim = hidden_dim_lstm * self.num_directions
        self.fc_inter = nn.Sequential(
            nn.Linear(lstm_output_dim + 4 + 64, hidden_inter),
            nn.LayerNorm(hidden_inter),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 5. لایه خروجی
        self.fc_out = nn.Linear(hidden_inter, output_size)
        
        # تابع فعال‌سازی
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()
        
        self._init_weights()
    
    def _init_weights(self):
        """مقداردهی اولیه"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        param.data.fill_(0)
    
    def forward(
        self,
        count_feats: torch.Tensor,      # [B, T, 7]
        in_time_feats: torch.Tensor,    # [B, T, 4]
        out_time_feats: torch.Tensor,   # [B, 4]
        text_feats: torch.Tensor,       # [B, T, 3072]
        hash_feats: torch.Tensor,       # [B, 1536]
        return_embedding: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: [batch_size, hidden_inter]
            output: [batch_size, 7]
        """
        # 1. پروجکشن متن و هشتگ
        text_proj = self.fc_text(text_feats)  # [B, T, hidden_text]
        hash_proj = self.fc_hashtag(hash_feats)  # [B, 64]
        
        # 2. ترکیب ویژگی‌های ورودی LSTM
        lstm_input = torch.cat([
            count_feats,
            text_proj,
            in_time_feats
        ], dim=-1)  # [B, T, 7+hidden_text+4]
        
        # 3. LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # 4. استخراج آخرین hidden state
        batch_size = count_feats.shape[0]
        
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            lstm_embedding = torch.cat([h_forward, h_backward], dim=-1)
        else:
            lstm_embedding = h_n[-1, :, :]
        
        # 5. ترکیب با ویژگی‌های دیگر
        combined = torch.cat([
            lstm_embedding,
            out_time_feats,
            hash_proj
        ], dim=-1)
        
        # 6. لایه میانی
        embedding = self.fc_inter(combined)
        
        # 7. خروجی نهایی
        output = self.fc_out(embedding)
        
        if return_embedding:
            return embedding, output
        return embedding, output


class BERTFineTuner(BaseForecaster):
    """
    مدل BERT برای Fine-tuning روی تسک NLP
    (برای مسیر NLP اجباری)
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,  # باینری یا چندکلاسه
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super(BERTFineTuner, self).__init__(output_size=num_classes)
        
        # بارگذاری BERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # فریز کردن لایه‌های BERT (اختیاری)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # طبقه‌بند
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # embedding_dim برای clustering
        self.embedding_dim = self.config.hidden_size
        
        self._init_weights()
    
    def _init_weights(self):
        """مقداردهی اولیه"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embedding: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: [batch_size, hidden_size]
            logits: [batch_size, num_classes]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # استفاده از [CLS] token
        pooled_output = outputs.pooler_output
        embedding = self.dropout(pooled_output)
        logits = self.classifier(embedding)
        
        if return_embedding:
            return embedding, logits
        return embedding, logits


def create_model(
    model_type: str,
    input_size: Optional[int] = None,
    **kwargs
) -> BaseForecaster:
    """
    کارخانه مدل برای ساخت آسان مدل‌ها
    
    Args:
        model_type: 'sfn_count', 'sfn_text', 'mfn', 'bert'
        input_size: فقط برای SFN
    """
    
    if model_type == 'sfn_count':
        return LSTMForecasterSF(
            input_size=7,
            hidden_dim_lstm=kwargs.get('hidden_dim_lstm', 128),
            num_layers=kwargs.get('num_layers', 2),
            bidirectional=kwargs.get('bidirectional', True),
            dropout=kwargs.get('dropout', 0.2)
        )
    
    elif model_type == 'sfn_text':
        return LSTMForecasterSF(
            input_size=3072,
            hidden_dim_lstm=kwargs.get('hidden_dim_lstm', 256),
            num_layers=kwargs.get('num_layers', 2),
            bidirectional=kwargs.get('bidirectional', True),
            dropout=kwargs.get('dropout', 0.3)
        )
    
    elif model_type == 'mfn':
        return LSTMForecasterMF(
            hidden_dim_lstm=kwargs.get('hidden_dim_lstm', 256),
            hidden_text=kwargs.get('hidden_text', 128),
            hidden_inter=kwargs.get('hidden_inter', 128),
            num_layers=kwargs.get('num_layers', 2),
            bidirectional=kwargs.get('bidirectional', True),
            dropout=kwargs.get('dropout', 0.2)
        )
    
    elif model_type == 'bert':
        return BERTFineTuner(
            model_name=kwargs.get('model_name', 'bert-base-uncased'),
            num_classes=kwargs.get('num_classes', 2),
            dropout=kwargs.get('dropout', 0.1),
            freeze_bert=kwargs.get('freeze_bert', False)
        )
    
    else:
        raise ValueError(f"Model type {model_type} not supported")


__all__ = [
    'BaseForecaster',
    'LSTMForecasterSF',
    'LSTMForecasterMF',
    'BERTFineTuner',
    'create_model'
]