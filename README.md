# Exoplanet Detection using Machine Learning

This repository implements an end-to-end pipeline for detecting exoplanets from light curve / Kepler transit data using machine learning classifiers.

## Project Structure

```text
exoplanet-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   └── 02_ml_classification.ipynb
│
├── src/
│   ├── feature_extraction.py
│   ├── train_classifier.py
│   └── evaluate.py
│
├── models/
│
├── results/
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── requirements.txt
└── README.md
```

## Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
