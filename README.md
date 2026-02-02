# Forecasting Political News Engagement on Social Media

This repository contains the data processing pipeline, feature extraction modules, modeling code, and analysis notebooks used in the paper:

**Forecasting Political News Engagement on Social Media**  
Karthik Shivaram, Mustafa Bilgic, Matthew A. Shapiro, Aron Culotta  
ICWSM 2024  

Paper link: https://ojs.aaai.org/index.php/ICWSM/article/view/31401

---

## Repository Overview

The project is organized to support end-to-end experimentation, from raw data inspection and preprocessing to model training, evaluation, and inference. All core functionality is implemented as reusable Python modules under `src/`, while execution and results are consolidated in a single Jupyter notebook.

**Key principle:** All major functions and classes defined in `src/` are used and demonstrated in **`main.ipynb`**, which serves as the primary entry point and produces the final experimental outputs.

---

## Data

### Data Access

The datasets used in this project are available at the following link:

https://www.dropbox.com/scl/fo/qpqzog2gl4r4w7c2j05k9/APWj4RonA6rPSJv1Zv0gS8Y?rlkey=7wr1x4slgeigsziyv1xvcgm0h&st=3po0hsz7&dl=0

Please download and place the data files in the appropriate data directory before running the notebook.

### Data Files

#### 1. `news_df.csv`
- Primary structured dataset containing political news articles and associated metadata.
- Serves as the main source for text content and numeric features.
- Used as the foundation for preprocessing, feature extraction, and modeling.

#### 2. `annotated_data_anonymized.jsonl`
- Contains labeled samples with engagement-related annotations.
- Used for supervised and semi-supervised learning experiments.
- Anonymized to remove personally identifiable information.

#### 3. `data_for_analysis/`
- Intermediate or derived datasets created during preprocessing and filtering.
- Used for statistical analysis and exploratory studies.

#### 4. `study_data/`
- Curated subsets of the data used for controlled experiments and evaluations.
- Helps clarify relationships between raw data, annotations, and modeling inputs.

---

## Source Code (`src/`)

All core logic is implemented in modular Python files under `src/`.

### Preprocessing

**`preprocessing_utils.py`**  
Text cleaning and normalization utilities, including:
- Tokenization and normalization
- Removal of noise and irrelevant symbols
- Preparation of text for feature extraction

### Dataset Construction

**`torch_datasets.py`**  
Defines PyTorch `Dataset` classes that:
- Wrap preprocessed text and numeric features
- Support batching and efficient loading during training and evaluation

### Feature Extraction

**`feature_extractors.py`**  
Implements text-based and numeric feature extraction methods, such as:
- Linguistic and lexical features
- Metadata-derived numerical features
- Combined feature representations

### Labeling

**`labelling_functions.py`**  
Labeling and weak supervision utilities, including:
- Heuristic labeling functions
- Rules for generating noisy or weak labels (when applicable)

### Filtering

**`filter_utils.py`**  
Functions for:
- Removing low-quality or invalid samples
- Applying data selection rules
- Creating clean and consistent training sets

### Analysis

**`analysis_utils.py`**  
Statistical analysis and evaluation helpers, including:
- Summary statistics
- Performance comparisons
- Error and distribution analysis

**`plot_utils.py`** *(optional at this stage)*  
Visualization utilities for:
- Exploratory data analysis
- Model performance plots
- Distribution and trend visualization

### Models

**`models.py`**  
Neural network model definitions, including:
- Architecture specifications
- Text and multimodal engagement prediction models

**`model_utils.py`**  
Training and evaluation utilities, such as:
- Training loops
- Validation and testing routines
- Metric computation and logging

### Inference

**`inference_utils.py`**  
Utilities for:
- Running trained models on new, unseen data
- Generating engagement predictions
- Post-processing model outputs

---

## Main Notebook

### `main.ipynb`

This notebook is the **central execution point** of the repository.

It:
- Loads and inspects all datasets
- Applies preprocessing and filtering
- Extracts features using `feature_extractors.py`
- Constructs PyTorch datasets via `torch_datasets.py`
- Trains and evaluates models defined in `models.py`
- Uses utilities from `model_utils.py`, `analysis_utils.py`, and `inference_utils.py`
- Produces all final experimental results and analyses

All functions defined in `src/` are invoked directly or indirectly within `main.ipynb`.

---

## Reproducibility

To reproduce the results:
1. Download the data from the provided Dropbox link.
2. Install the required Python dependencies.
3. Run `main.ipynb` end-to-end.

The final outputs, including trained models, evaluation metrics, and analyses, are generated within the notebook.
