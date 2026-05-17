# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview
This project builds and analyzes a feed-forward neural network for binary classification — predicting customer churn from structured tabular data.

## Dataset
**Source:** [BITSoM BA Module 5 Dataset](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
**File:** `customer_churn_nn.csv`
**Features:** Age, tenure, monthly charges, total charges, contract type, payment method, support calls, satisfaction score, product subscriptions
**Target:** `churn` (0 = No, 1 = Yes)

## Approach

### Task 1 – Dataset Understanding
- Explored 1000 rows × 12 columns
- Target: binary churn label (~40% positive rate)
- No missing values found
- Key insights: month-to-month contracts and low satisfaction scores correlate with higher churn

### Task 2 – Data Preprocessing
- Dropped non-informative customer_id column
- Label-encoded categorical features (contract_type, payment_method)
- Applied StandardScaler on numerical features
- 80/20 stratified train-test split

### Task 3 – Neural Network Architecture
- Input layer → 2 × Dense(64, ReLU) + Dropout(0.2) → Dense(1, sigmoid)
- Loss: Binary Cross-Entropy | Optimizer: Adam | Metric: Accuracy

### Task 4 – Training & Evaluation
- Trained for 50 epochs, batch size 32
- Evaluated with accuracy, classification report, and confusion matrix

### Task 5 – Hyperparameter Experiments (6 configurations)
| Experiment | Train Acc | Test Acc |
|---|---|---|
| Baseline (64 units, 2 layers, relu, lr=0.001) | ~0.70 | ~0.67 |
| More Neurons (128 units) | ~0.73 | ~0.68 |
| Deeper Network (3 layers) | ~0.72 | ~0.68 |
| Lower LR (0.0001) | ~0.65 | ~0.63 |
| Tanh Activation | ~0.70 | ~0.66 |
| Larger Batch (128) | ~0.69 | ~0.66 |

### Task 6 – Final Reflection
- Weights & Biases: Learned parameters updated via backpropagation
- Activation Functions: Introduce non-linearity
- Learning Rate: Too high diverges; too low converges slowly
- Overfitting/Underfitting: Mild underfitting in baseline; deeper networks helped

## Results
- Best test accuracy: ~68% (128 neurons / 3-layer configurations)

## Requirements
See `requirements.txt` for all dependencies.