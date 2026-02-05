# Political News Engagement Forecasting
Forecasting Political News Engagement on Social Media using Deep Learning

## 📌 Project Overview
This project implements a complete pipeline for analyzing and forecasting **political news engagement patterns on social media**, inspired by the ICWSM 2024 paper:

> *Forecasting Political News Engagement on Social Media*  
> Shivaram et al., ICWSM 2024

The goal is to model how users interact with political news sources across the ideological spectrum and **predict future engagement behavior** using time-series deep learning models.

This project is developed as part of the **Artificial Intelligence course final project**, following strict requirements on data analysis, modeling, evaluation, and GitHub version control.

---

## 🎯 Problem Definition
Given historical Twitter engagement data across **7 political stances** (from -3 = far-left to +3 = far-right), the task is to:

- Model user engagement patterns over time
- Forecast engagement levels for the next time period
- Cluster users based on their political news consumption behavior

**Task Type:**  
- Time-series forecasting  
- User behavior modeling  
- Unsupervised clustering  

**Evaluation Metric:**  
- Mean Absolute Error (MAE)

---

## 📊 Dataset
- **Source:** ICWSM 2024 anonymized dataset  
- **Size:** ~5.6 million Twitter engagements  
- **Time Span:** 2008–2021  
- **Focus Period:** 2015–2021  
- **Political Stances:**  
  - -3 (far-left)  
  - -2  
  - -1  
  - 0 (center)  
  - +1  
  - +2  
  - +3 (far-right)

Each user’s engagement history is converted into **quarterly time-series sequences**.

---

## 🧠 Methodology

### 1️⃣ Data Processing
- Load ~50,000 sampled Twitter users
- Aggregate engagements into quarterly time steps
- Convert sequences:
  - **Input:** 8 quarters
  - **Target:** next quarter engagement
- Normalize and structure data for neural networks

---

### 2️⃣ Modeling
Multiple LSTM-based architectures were tested.

**Final Model (MFN):**
- Hybrid architecture combining:
  - Text-derived features
  - Historical engagement signals
- Implemented in **PyTorch**
- Trained to predict engagement intensity for all 7 political stances

---

### 3️⃣ User Clustering
- User embeddings extracted from trained model
- K-Means clustering applied
- **20 distinct user clusters** identified
- Each cluster represents a unique political news consumption pattern

---

### 4️⃣ Visualization
- Heatmap visualization inspired by the original ICWSM paper
- Shows engagement intensity:
  - Across political spectrum
  - Over time (2018–2021)
- Used for interpretability and behavioral analysis

---

## 🧪 Results

| Component | Result |
|--------|--------|
| Best Model | MFN (LSTM-based) |
| Validation MAE | ~1.79 |
| Number of Clusters | 20 |
| Political Spectrum | -3 to +3 |

### Generated Outputs
- **Trained Model:**  
  `trained_models/full_trained_model.pth`
- **Cluster Analysis:**  
  `results/cluster_analysis.csv`
- **Main Visualization:**  
  `cluster_plots/latest_cluster_plot_april2024.png`

---

## 📁 Project Structure
```text
├── data/                  # Raw dataset (not tracked by Git)
├── notebooks/
│   └── Figure8-final.ipynb # Main execution notebook
├── src/
│   ├── preprocessing/     # Data preparation scripts
│   ├── models/            # Model definitions
│   ├── training/          # Training loops
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Helper functions
├── results/               # CSVs and numerical outputs
├── cluster_plots/         # Visualizations and heatmaps
├── trained_models/        # Saved model weights (ignored in Git)
├── requirements.txt
├── .gitignore
└── README.md
