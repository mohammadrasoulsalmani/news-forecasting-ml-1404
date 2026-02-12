# 🗞️ Political News Engagement Forecasting on Twitter

## 📌 Project Overview
Deep learning pipeline for forecasting user engagement with political news on Twitter across **7 ideological stances** (-3 far-left to +3 far-right).

- **Task:** Time-series forecasting + User clustering  
- **Evaluation Metric:** Mean Absolute Error (MAE)

---

## 📊 Dataset

| Item | Value |
|------|--------|
| Source | ICWSM 2024 (anonymized) |
| Sample | 10% (563,778 records) |
| Users | 5,975 unique |
| Time Span | 2009–2021 |
| Sequences | 39,044 (8 quarters → 1 quarter) |
| Input Shape | (batch, 8, 7) |
| Target Shape | (batch, 7) |

**Political Stances:** `-3, -2, -1, 0, +1, +2, +3`

---

## 🧠 Methodology

### 1️⃣ Data Processing
- Random sampling (10%)
- Quarterly aggregation
- 80/20 train-validation split

---

### 2️⃣ Models

| Model | Performance |
|--------|-------------|
| Baseline (Last Value) | MAE: 3.89 |
| Logistic Regression + TF-IDF | Accuracy: 68% |
| Bidirectional LSTM | **MAE: 3.73 (+4.1%)** |

#### LSTM Architecture
- 2 layers
- 128 hidden units
- Bidirectional
- Dropout: 0.3
- Optimizer: Adam (`lr=1e-3`)
- Early stopping (`patience=5`)

---

### 3️⃣ User Clustering
- Extracted **256-dim embeddings** from LSTM hidden state
- K-Means clustering (`K=10`)
- 7,809 validation sequences embedded

#### Top Clusters

| Cluster | Users | Avg Stance |
|----------|--------|-------------|
| 2 | 2,106 (27%) | -0.45 |
| 4 | 1,547 (20%) | -0.12 |
| 8 | 1,452 (19%) | -0.93 |

---

## 📁 Project Structure

```text
├── config/               # Hyperparameters
├── dataLoader/           # JSON loading + sequence builder
├── models/               # Bidirectional LSTM
├── training/             # Trainer + early stopping
├── evaluation/           # Clustering + metrics
├── visualization/        # Plots (curves, heatmaps)
├── notebooks/            # EDA, baseline, experiments
├── scripts/              # train_pipeline.py
├── models_saved/         # full_model.pth (2.1 MB)
└── results/              # cluster_results.pkl + figures
```

## 🚀 Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

```bash
cd scripts
python train_pipeline.py
```



📊 Version: 1.0.0
🎓 Course: AI
👨‍🏫 Instructor: Dr. Pishgoo
🧑‍💻 TA: Eng. Ghorbani