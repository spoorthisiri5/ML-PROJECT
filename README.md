# 🍝 Spaghetti Code Detector

> An "Angry Senior Dev AI" that predicts software bugs using the NASA JM1 defect dataset and machine learning.

## What it does

Input your code module's metrics using sliders. Click **Submit for Code Review**. A RandomForest model trained on ~10,000 real NASA software modules predicts whether your code is likely to contain defects — and delivers the verdict in the voice of a highly critical Senior Developer.

## Tech stack

| Layer | Technology |
|---|---|
| Dataset | NASA MDP JM1 (McCabe & Halstead metrics) |
| ML model | RandomForestClassifier + LogisticRegression |
| Framework | Scikit-learn |
| UI | Streamlit |
| Language | Python 3.10+ |

## Project structure

```
spaghetti-detector/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── model.pkl               # Trained RandomForest
├── lr_model.pkl            # Trained LogisticRegression
├── scaler.pkl              # Fitted StandardScaler
├── feature_names.pkl       # Ordered feature column list
└── notebook/
    └── train_model.ipynb   # Full training notebook
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Training the model

Open `notebook/train_model.ipynb` in Google Colab or Jupyter.  
Download the [NASA JM1 dataset from Kaggle](https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction) and place `jm1.csv` in the notebook folder.  
Run all cells — the notebook exports the four `.pkl` files automatically.

## Features

- Predicts defect probability using 21 McCabe & Halstead software metrics
- Spaghetti-o-meter progress bar showing bug risk level
- Feature importance chart explaining the model's decision
- Specific advice for each metric that exceeds safe thresholds
- Sidebar model toggle (RandomForest vs Logistic Regression)
- Senior Dev mood selector (affects response flavour)

## Dataset

NASA Metrics Data Program — JM1  
~10,885 software modules from a real-time predictive ground system written in C.  
Target: `defects` (True = module contained at least one reported bug)  
Defect rate: ~19%

---

*Built for academic demonstration of ML applied to software engineering.*
