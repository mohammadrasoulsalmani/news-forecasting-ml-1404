"""
کلاس Trainer برای مدیریت چرخه آموزش.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from config.settings import Config
from utils.helpers import EarlyStopping

class Trainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion=nn.L1Loss()
    ):
        self.config = config
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.early_stopping = EarlyStopping(patience=config.PATIENCE, verbose=False)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_seq, batch_label in self.train_loader:
            batch_seq = batch_seq.to(self.config.DEVICE)
            batch_label = batch_label.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            _, pred = self.model(batch_seq)
            loss = self.criterion(pred, batch_label)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_seq, batch_label in self.val_loader:
                batch_seq = batch_seq.to(self.config.DEVICE)
                batch_label = batch_label.to(self.config.DEVICE)
                _, pred = self.model(batch_seq)
                loss = self.criterion(pred, batch_label)
                total_loss += loss.item()
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch_label.cpu().numpy())
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        return total_loss / len(self.val_loader), all_preds, all_labels
    
    def fit(self):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch()
            val_loss, val_preds, val_labels = self.validate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1:2d}/{self.config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("⚠️ Early stopping triggered.")
                break
        
        # بارگذاری بهترین checkpoint
        self.model.load_state_dict(torch.load(self.early_stopping.save_path))
        return train_losses, val_losses, val_preds, val_labels