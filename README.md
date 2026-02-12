# 🗞️ Political News Engagement Forecasting on Twitter
Forecasting user engagement with political news on Twitter using deep learning and time-series modeling.

---

## 📌 Project Overview
This project implements a **deep-learning pipeline for forecasting political news engagement patterns on Twitter**, leveraging longitudinal user interaction data across the political spectrum.  

The goal is to **predict future user engagement per ideological stance** while uncovering behavioral segments through clustering.  

Developed as part of the **Artificial Intelligence course final project**, focusing on reproducibility, evaluation, and interpretability.

---

## 🎯 Problem Definition
Given historical Twitter engagement data categorized by **7 political stances** (-3 = far-left, 0 = center, +3 = far-right), the task is to:

- Model longitudinal engagement patterns of users
- Forecast quarterly engagement counts across ideological bins
- Segment users based on their news consumption behaviors

**Task Types:**  
- Time-series forecasting  
- User behavior modeling  
- Unsupervised clustering  

**Evaluation Metric:**  
- Mean Absolute Error (MAE)

---

## 📊 Dataset
- **Source:** ICWSM 2024 anonymized Twitter dataset  
- **Size:** ~50,000 sampled users  
- **Time Span:** 2008–2021 (focus 2015–2021)  
- **Political Stances:**  
  - -3 (far-left)  
  - -2  
  - -1  
  - 0 (center)  
  - +1  
  - +2  
  - +3 (far-right)  

Each user’s engagement history is aggregated into **quarterly time-series sequences**.

---

## 🧠 Methodology

### 1️⃣ Data Processing
- Load and sample user engagement histories
- Aggregate engagements into quarterly bins
- Convert sequences into neural network inputs:
  - **Input:** last 8 quarters  
  - **Target:** next quarter engagement
- Normalize and structure data for deep learning models

### 2️⃣ Modeling
- Logistic Regression + TF-IDF baseline
- **Final Model:** Bidirectional LSTM (transformer variant available)  
- Trained in **PyTorch** to predict engagement intensity across 7 ideological stances

### 3️⃣ User Clustering
- Extract user embeddings from trained model
- Apply **K-Means clustering**
- Identify **20 behavioral clusters**
- Each cluster represents a distinct political news consumption pattern

### 4️⃣ Visualization
- Heatmaps and training curves
- Display engagement across political spectrum over time
- Support interpretability and cluster analysis

---

## 🧪 Results

| Component | Result |
|-----------|--------|
| Best Model | BiLSTM |
| Validation MAE | ~3.73 (baseline: 3.89) |
| Number of Clusters | 20 |
| Political Spectrum | -3 to +3 |

### Generated Outputs
- **Trained Model:**  
  `models_saved/full_model.pth`
- **Cluster Analysis:**  
  `results/cluster_analysis.csv`
- **Visualizations:**  
  `results/figures/` (heatmaps & training curves)

---

## 📁 Project Structure
```text
├── config/               # Configuration and hyperparameters
├── data/                 # Raw & preprocessed data (not tracked by Git)
├── models/               # LSTM & Transformer model definitions
├── training/             # Training loops and trainer scripts
├── evaluation/           # Metrics computation and cluster analysis
├── visualization/        # Plotting utilities & figures
├── notebooks/            # EDA and experiments
├── scripts/              # CLI pipelines (train, evaluate, cluster)
├── models_saved/         # Saved PyTorch model weights (ignored in Git)
├── results/              # Metrics, CSVs, figures
├── requirements.txt
├── README.md
└── .gitignore
