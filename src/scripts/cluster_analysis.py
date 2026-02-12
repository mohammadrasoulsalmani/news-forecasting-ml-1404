#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
بارگذاری مدل آموزش‌دیده، استخراج embedding و خوشه‌بندی.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from config.settings import Config
from models.forecaster import NewsForecaster
from evaluation.clustering import ClusterAnalyzer
from visualization.plots import Visualizer
from utils.term_extraction import extract_important_terms

def main():
    config = Config()
    model_path = config.MODEL_SAVE_DIR / "full_model.pth"
    
    # بارگذاری مدل
    model = NewsForecaster(config)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    
    # بارگذاری داده (یا استفاده از داده ذخیره شده)
    # اینجا برای سادگی فرض می‌کنیم داده val از قبل وجود دارد
    # در عمل بهتر است داده‌های اعتبارسنجی ذخیره شوند.
    # ...
    
    # استخراج hidden و خوشه‌بندی
    analyzer = ClusterAnalyzer(config)
    viz = Visualizer(config)
    
    # ... (مشابه train_pipeline)
    
if __name__ == "__main__":
    main()