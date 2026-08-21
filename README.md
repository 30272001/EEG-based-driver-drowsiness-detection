# EEG-based-driver-drowsiness-detection
Machine learning system for real-time driver drowsiness detection using multi-channel EEG signals, SMOTE class balancing, and Random Forest classification.

# Multi-Channel EEG and Machine Learning for Optimized Driver Drowsiness Detection

## Overview
Drowsy driving significantly reduces reaction time and increases road accident risks. This project implements an automated machine learning system using EEG Dataset from physiological signals collected via a NeuroSky MindWave headset to detect driver fatigue in real time.

## Dataset Details
- **Instances:** 3,735 samples collected across 4 drivers
- **Features (11):** `Attention`, `Meditation`, `Delta`, `Theta`, `Low Alpha`, `High Alpha`, `Low Beta`, `High Beta`, `Low Gamma`, `High Gamma`
- **Target:** Binary Classification (`0`: Drowsy, `1`: Alert)

## Methodology Pipeline
1. **Preprocessing:** Missing value handling, complete-case analysis, and feature scaling using `MinMaxScaler`.
2. **Class Balancing:** Synthetic Minority Over-sampling Technique (SMOTE) applied to eliminate minority class bias:
   $$x_{synthetic} = x_i + \lambda \cdot (x_{nn} - x_i)$$
3. **Model Selection & Tuning:** Hyperparameter optimization via `GridSearchCV` using 5-Fold Stratified Cross-Validation.

## Experimental Results
Multiple classical machine learning algorithms were trained and evaluated on test data:

| Model               | Test Accuracy | Precision (Alert / Drowsy) | Recall (Alert / Drowsy) |   F1-Score  |
------------------------------------------------------------------------------------------------------------
| Random Forest       |    78.41%     |        0.81 / 0.75         |       0.81 / 0.75       | 0.81 / 0.75 |
------------------------------------------------------------------------------------------------------------
| Decision Tree       |    71.99%     |        0.76 / 0.67         |       0.74 / 0.69       | 0.75 / 0.68 |
------------------------------------------------------------------------------------------------------------
| K-Nearest Neighbors |    71.54%     |        0.78 / 0.65         |       0.69 / 0.74       | 0.74 / 0.69 |
------------------------------------------------------------------------------------------------------------
| Logistic Regression |    68.51%     |        0.79 / 0.60         |       0.61 / 0.78       | 0.69 / 0.68 |
------------------------------------------------------------------------------------------------------------

## Random Forest achieved peak training accuracy of 81.92% and overall test accuracy of 78.41% with optimal precision-recall balance.
