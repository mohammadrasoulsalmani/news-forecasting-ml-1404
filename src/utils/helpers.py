"""
توابع کمکی: EarlyStopping, تنظیم seed, ...
"""

import torch
import numpy as np
import random

class EarlyStopping:
    """Early stopping با ذخیره بهترین مدل."""
    def __init__(self, patience=5, delta=0.001, verbose=False, save_path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"Validation loss decreased. Model saved.")


def set_seed(seed):
    """تنظیم seed برای بازتولیدپذیری."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False