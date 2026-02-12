#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Pipeline for Political News Engagement Forecasting.
Complete pipeline with detailed progress tracking.
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from config.settings import Config
from dataLoader.loader import DataLoader as NewsDataLoader
from dataLoader.preprocessor import SequenceBuilder
from models.forecaster import NewsForecaster
from training.trainer import Trainer
from evaluation.clustering import ClusterAnalyzer
from visualization.plots import Visualizer
from utils.helpers import set_seed


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*60)
    print(f"📌 {title}")
    print("="*60)


def main():
    """Execute the complete training and analysis pipeline."""
    
    # Initialize configuration and set random seed
    config = Config()
    set_seed(config.RANDOM_SEED)
    
    print("="*60)
    print("📰 POLITICAL NEWS ENGAGEMENT FORECASTING")
    print("="*60)
    print(f"📊 Configuration:")
    print(f"   • Device: {config.DEVICE}")
    print(f"   • Sample fraction: {config.SAMPLE_FRACTION:.0%}")
    print(f"   • Sequence length: {config.SEQ_LENGTH}")
    print(f"   • Hidden dim: {config.HIDDEN_DIM}")
    print(f"   • Batch size: {config.BATCH_SIZE}")
    print("="*60)
    
    # ---------- 1. Data Loading ----------
    print_section("DATA LOADING")
    start_time = time.time()
    
    loader = NewsDataLoader(config)
    df = loader.load(sample_fraction=config.SAMPLE_FRACTION)
    
    load_time = time.time() - start_time
    print(f"⏱️  Data loading completed in {load_time:.1f} seconds")
    
    # ---------- 2. Sequence Building ----------
    print_section("SEQUENCE BUILDING")
    start_time = time.time()
    
    builder = SequenceBuilder(config)
    sequences, labels = builder.build(df)
    
    build_time = time.time() - start_time
    print(f"⏱️  Sequence building completed in {build_time:.1f} seconds")
    print(f"📊 Final dataset: {len(sequences):,} sequences, shape {sequences.shape}")
    
    # ---------- 3. Train/Validation Split ----------
    print_section("DATA SPLIT")
    split = int(len(sequences) * (1 - config.TEST_SPLIT))
    X_train, X_val = sequences[:split], sequences[split:]
    y_train, y_val = labels[:split], labels[split:]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"   ✅ Training set: {len(X_train):,} sequences ({len(X_train)/len(sequences):.1%})")
    print(f"   ✅ Validation set: {len(X_val):,} sequences ({len(X_val)/len(sequences):.1%})")
    print(f"   📦 Training batches: {len(train_loader)}")
    print(f"   📦 Validation batches: {len(val_loader)}")
    
    # ---------- 4. Model Training ----------
    print_section("MODEL TRAINING")
    
    model = NewsForecaster(config)
    print(f"   🧠 Model architecture:")
    print(f"      • LSTM: {config.NUM_LAYERS} layers, {config.HIDDEN_DIM} units")
    print(f"      • Bidirectional: {config.BIDIRECTIONAL}")
    print(f"      • Dropout: {config.DROPOUT}")
    print(f"      • Output: {config.NUM_STANCES} stances")
    
    trainer = Trainer(config, model, train_loader, val_loader)
    train_losses, val_losses, val_preds, val_labels = trainer.fit()
    
    # ---------- 5. Model Saving ----------
    print_section("MODEL SAVING")
    model_path = config.MODEL_SAVE_DIR / "full_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"   💾 Model saved to {model_path}")
    print(f"   📦 Model size: {Path(model_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    # ---------- 6. User Clustering ----------
    print_section("USER CLUSTERING")
    
    analyzer = ClusterAnalyzer(config)
    
    # Step 6.1: Extract hidden states
    print("   📥 Step 1/3: Extracting hidden states from LSTM...")
    start_time = time.time()
    hidden = analyzer.extract_hidden_states(model, val_loader)
    extract_time = time.time() - start_time
    print(f"      ✅ Extracted {len(hidden):,} embeddings with shape {hidden.shape}")
    print(f"      ⏱️  Time: {extract_time:.1f} seconds")
    
    # Step 6.2: K-means clustering
    print(f"\n   🔍 Step 2/3: Running K-means with {config.NUM_CLUSTERS} clusters...")
    start_time = time.time()
    cluster_labels = analyzer.cluster_users(hidden)
    cluster_time = time.time() - start_time
    print(f"      ✅ Clustering completed")
    print(f"      ⏱️  Time: {cluster_time:.1f} seconds")
    
    # Analyze cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"      📊 Cluster size distribution:")
    for cid, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5]:
        print(f"         • Cluster {cid}: {count:,} users ({count/len(cluster_labels):.1%})")
    
    # Step 6.3: Analyze cluster engagements
    print(f"\n   📊 Step 3/3: Analyzing cluster engagement patterns...")
    start_time = time.time()
    cluster_data = analyzer.analyze_cluster_engagements(cluster_labels, val_labels)
    analyze_time = time.time() - start_time
    print(f"      ✅ Analyzed {len(cluster_data)} clusters")
    print(f"      ⏱️  Time: {analyze_time:.1f} seconds")
    
    # Show cluster stances
    print(f"\n   📈 Cluster political stances:")
    sorted_clusters = sorted(cluster_data.items(), key=lambda x: x[1]['avg_stance'])
    for cid, data in sorted_clusters[:3]:
        stance_desc = "Liberal" if data['avg_stance'] < -0.5 else "Centrist" if abs(data['avg_stance']) < 0.5 else "Conservative"
        print(f"      • Cluster {cid}: {data['size']} users, avg stance: {data['avg_stance']:.2f} ({stance_desc})")
    print("      • ...")
    for cid, data in sorted_clusters[-3:]:
        stance_desc = "Liberal" if data['avg_stance'] < -0.5 else "Centrist" if abs(data['avg_stance']) < 0.5 else "Conservative"
        print(f"      • Cluster {cid}: {data['size']} users, avg stance: {data['avg_stance']:.2f} ({stance_desc})")
    
    total_cluster_time = extract_time + cluster_time + analyze_time
    print(f"\n   ⏱️  Total clustering time: {total_cluster_time:.1f} seconds")
    
    # ---------- 7. Visualization ----------
    print_section("VISUALIZATION")
    
    viz = Visualizer(config)
    
    print("   🎨 Plot 1/2: Training curves...")
    viz.plot_training_curves(train_losses, val_losses)
    
    print("   🎨 Plot 2/2: Cluster heatmap...")
    viz.plot_cluster_heatmap(cluster_data)
    
    # ---------- 8. Save Results ----------
    print_section("SAVING RESULTS")
    
    results_path = config.RESULT_DIR / "cluster_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({
            'cluster_data': cluster_data,
            'cluster_labels': cluster_labels,
            'hidden_states': hidden,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_predictions': val_preds,
            'val_labels': val_labels,
            'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        }, f)
    
    print(f"   💾 Results saved to {results_path}")
    print(f"   📦 File size: {results_path.stat().st_size / 1024:.1f} KB")
    
    # ---------- 9. Summary ----------
    print_section("PIPELINE COMPLETED")
    print(f"   ✅ Status: SUCCESS")
    print(f"   📁 Output directory: {config.RESULT_DIR}")
    print(f"   • Model: {model_path.name}")
    print(f"   • Results: {results_path.name}")
    print(f"   • Figures: {config.FIGURE_DIR}")
    print(f"\n   📊 Final statistics:")
    print(f"      • Users analyzed: {df['user_id'].nunique():,}")
    print(f"      • Sequences created: {len(sequences):,}")
    print(f"      • Clusters discovered: {len(cluster_data)}")
    print(f"      • Model MAE: {np.mean(np.abs(val_preds - val_labels)):.4f}")
    
    print("\n" + "="*60)
    print("🎯 PIPELINE EXECUTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()