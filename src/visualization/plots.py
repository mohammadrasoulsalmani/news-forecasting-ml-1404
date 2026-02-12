"""
توابع بصری‌سازی: منحنی آموزش، هیت‌مپ خوشه‌ها، ...
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config.settings import Config

class Visualizer:
    def __init__(self, config: Config):
        self.config = config
    
    def plot_training_curves(self, train_losses, val_losses, save_name="training_curves.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MAE Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = self.config.FIGURE_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Saved: {save_path}")
    
    def plot_cluster_heatmap(self, cluster_data: dict, save_name="cluster_heatmap.png"):
        """هیت‌مپ ساده از الگوی تعامل هر خوشه."""
        n_clusters = len(cluster_data)
        engagement_matrix = np.zeros((n_clusters, 7))
        sizes = []
        for cid, data in sorted(cluster_data.items()):
            engagement_matrix[cid] = data['engagement_pattern']
            sizes.append(data['size'])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(engagement_matrix, 
                   annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=['-3','-2','-1','0','+1','+2','+3'],
                   yticklabels=[f'Cluster {cid}\n(n={sizes[cid]})' 
                               for cid in sorted(cluster_data.keys())])
        plt.title('Average News Engagement by Cluster')
        plt.xlabel('Political Stance')
        plt.ylabel('Cluster')
        plt.tight_layout()
        save_path = self.config.FIGURE_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Saved: {save_path}")
    
    def plot_advanced_clusters(self, cluster_data: dict, 
                               important_terms: dict = None,
                               save_name="advanced_clusters.png"):
        """
        نسخه پیشرفته مشابه شکل ۵ مقاله.
        نیاز به داده‌های زمانی بیشتر دارد؛ در اینجا با تکرار ۴ ساله شبیه‌سازی شده.
        """
        n_clusters = len(cluster_data)
        rows = 5
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(21, 18))
        axes = axes.ravel()
        
        # مرتب‌سازی بر اساس میانگین گرایش
        sorted_clusters = sorted(cluster_data.keys(), 
                                key=lambda c: cluster_data[c]['avg_stance'])
        
        for idx, cid in enumerate(sorted_clusters):
            if idx >= len(axes):
                break
            ax = axes[idx]
            data = cluster_data[cid]
            pattern = data['engagement_pattern']
            # تبدیل به ماتریس 7×4 (تکرار برای ۴ سال)
            matrix = np.tile(pattern, (4, 1)).T
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', 
                          vmin=0, vmax=np.max(pattern)*1.2)
            
            # عنوان: شماره خوشه، درصد، کلمات مهم
            title = f"#{cid+1} ({data['size']} users)"
            if important_terms and cid in important_terms:
                terms = important_terms[cid][:3]
                title += "\n" + ", ".join(terms)
            ax.set_title(title, fontsize=9)
            
            if idx % cols == 0:
                ax.set_ylabel('Stance', fontsize=9)
                ax.set_yticks(range(7))
                ax.set_yticklabels(['-3','-2','-1','0','+1','+2','+3'])
            else:
                ax.set_yticks([])
            
            if idx >= (rows-1)*cols:
                ax.set_xlabel('Year', fontsize=9)
                ax.set_xticks(range(4))
                ax.set_xticklabels(['2018','2019','2020','2021'])
            else:
                ax.set_xticks([])
        
        # مخفی کردن زیرنمودارهای اضافی
        for idx in range(len(sorted_clusters), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('User Clusters by News Engagement Patterns', fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        save_path = self.config.FIGURE_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Saved advanced cluster plot: {save_path}")