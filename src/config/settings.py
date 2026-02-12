"""
تنظیمات و هایپرپارامترهای پروژه.
همه مقادیر قابل تغییر در اینجا متمرکز شده‌اند.
"""

import torch
from pathlib import Path

class Config:
    # -------------------- XXXXXXX --------------------
    DATA_PATH = Path("../data/icwsm-2024-forecasting-data-anon.json")  # برای اجرا از scripts/
    
    MODEL_SAVE_DIR = Path("models_saved")
    RESULT_DIR = Path("results")
    FIGURE_DIR = RESULT_DIR / "figures"
    PREDICTION_DIR = RESULT_DIR / "predictions"
    
    # -------------------- Data Settings --------------------
    # Sampling - 10% default for quick experimentation
    SAMPLE_FRACTION = 0.10  # 10% of data (recommended)
    SAMPLE_SIZE = None      # Set this to override fraction (e.g., 50000)
    
    # Sequence parameters
    SEQ_LENGTH = 8
    NUM_STANCES = 7
    TEST_SPLIT = 0.2
    
    # -------------------- Model Architecture --------------------
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True
    
    # -------------------- Training --------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    PATIENCE = 5
    RANDOM_SEED = 42
    
    # -------------------- Clustering --------------------
    NUM_CLUSTERS = 10
    
    # -------------------- Device --------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        """Create necessary directories."""
        self.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        self.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        self.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Verify data path
        if not self.DATA_PATH.exists():
            print(f"⚠️  Data file not found: {self.DATA_PATH}")
            print(f"📁 Current directory: {Path.cwd()}")
            print(f"📁 Project root: {self.BASE_DIR}")