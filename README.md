# Customer Churn Prediction with ANN

A deep learning project that predicts whether a bank customer will churn (leave) using an Artificial Neural Network, with an interactive Streamlit web app for real-time predictions.

## Overview

Customer churn is a critical metric for banks. This project builds a binary classifier using a feedforward ANN trained on customer demographics and account data to predict churn with ~86% accuracy.

## Project Structure

```
├── classification.ipynb   # Main notebook: data preprocessing, ANN training, artifact export
├── regression.ipynb       # Bonus notebook: ANN regression to predict EstimatedSalary
├── prediction.ipynb       # Inference demo using the trained classification model
├── app.py                 # Streamlit web app for interactive churn prediction
├── Churn_Modelling.csv    # Dataset (10,000 bank customers)
├── train.png              # Training accuracy/loss plot
└── requirements.txt       # Pinned dependencies
```

## Dataset

The [Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) contains 10,000 records with 13 features:

| Feature | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is active |
| EstimatedSalary | Estimated annual salary |
| **Exited** | **Target: 1 = churned, 0 = retained** |

## Model Architecture

```
Input (12 features)
    ↓
Dense(64, ReLU)
    ↓
Dense(32, ReLU)
    ↓
Dense(1, Sigmoid)  →  Churn probability
```

- **Optimizer:** Adam (lr=0.01)
- **Loss:** Binary Cross-Entropy
- **Callbacks:** EarlyStopping (patience=10) + TensorBoard
- **Test Accuracy:** ~86%

## Setup

**1. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3/M4/M5):** `tensorflow-metal` is included in `requirements.txt` and enables GPU acceleration via Metal. No extra steps needed — TensorFlow will automatically route training to the GPU.

**3. Train the model and generate artifacts:**

Open and run all cells in `classification.ipynb`. This will produce:
- `model.keras` — trained ANN
- `scaler.pkl` — StandardScaler
- `label_encoder.pkl` — LabelEncoder for Gender
- `one_hot_encoder.pkl` — OneHotEncoder for Geography

## Running the App

```bash
streamlit run app.py
```

The app lets you adjust customer attributes via sliders and dropdowns and instantly shows the predicted churn probability.

## TensorBoard

Training logs are saved to the `logs/fit/` directory. To visualize:
```bash
tensorboard --logdir logs/fit
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `classification.ipynb` | Full pipeline: EDA → preprocessing → ANN training → artifact export |
| `regression.ipynb` | ANN regression experiment predicting EstimatedSalary |
| `prediction.ipynb` | Standalone inference demo on a single customer record |


## Author

Built by **[Karan Khatavkar](https://karankh.tech)** — AI/ML engineer (LLM, RAG & NLP), open to freelance & contract work.
