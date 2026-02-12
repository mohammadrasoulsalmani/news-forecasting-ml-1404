"""
خوشه‌بندی کاربران بر اساس embedding استخراج شده از LSTM.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from config.settings import Config

class ClusterAnalyzer:
    def __init__(self, config: Config):
        self.config = config
    
    def extract_hidden_states(self, model, dataloader: DataLoader) -> np.ndarray:
        """استخراج embedding لایه پنهان برای همه نمونه‌های مجموعه."""
        model.eval()
        hidden_states = []
        with torch.no_grad():
            for batch_seq, _ in dataloader:
                batch_seq = batch_seq.to(self.config.DEVICE)
                embedding, _ = model(batch_seq, return_embedding=True)
                hidden_states.append(embedding.cpu().numpy())
        return np.vstack(hidden_states)
    
    def cluster_users(self, representations: np.ndarray) -> np.ndarray:
        """K-means با تعداد خوشه از پیش تعیین‌شده."""
        kmeans = KMeans(
            n_clusters=self.config.NUM_CLUSTERS,
            random_state=self.config.RANDOM_SEED,
            n_init=10
        )
        return kmeans.fit_predict(representations)
    
    @staticmethod
    def analyze_cluster_engagements(cluster_labels: np.ndarray, 
                                    user_labels: np.ndarray) -> dict:
        """
        محاسبه میانگین تعاملات و اندازه هر خوشه.
        cluster_labels: (n_samples,) برچسب خوشه
        user_labels  : (n_samples, 7) بردار تعاملات واقعی
        """
        n_clusters = len(np.unique(cluster_labels))
        cluster_data = {}
        for cid in range(n_clusters):
            indices = np.where(cluster_labels == cid)[0]
            if len(indices) > 0:
                avg_engagement = np.mean(user_labels[indices], axis=0)
                # میانگین وزنی گرایش سیاسی
                stances = np.array([-3, -2, -1, 0, 1, 2, 3])
                total = np.sum(avg_engagement)
                avg_stance = np.sum(stances * avg_engagement) / total if total > 0 else 0.0
                cluster_data[cid] = {
                    'size': len(indices),
                    'avg_stance': avg_stance,
                    'engagement_pattern': avg_engagement
                }
        return cluster_data