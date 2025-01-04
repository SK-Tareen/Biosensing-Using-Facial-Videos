# Blood Pressure Prediction Using Ensemble Machine Learning Models

## Overview

This repository was my final year project in Bachelors. It contains a Python-based project for predicting systolic and diastolic blood pressure using ensemble learning techniques, including Random Forest and Gradient Boosting Regressors. The model incorporates advanced signal processing, feature engineering, and hyperparameter tuning for robust prediction.
We used the Vital Videos dataset, because it is one of the few that we could use which includes blood pressure with heartbeat. 
The Dataloader.csv file is the result of compiling the ground truth data from the data given in the dataset. The testing was done based on results we got from the videos when we applied our models on them.

## Features

- **Data Preprocessing**: Cleansing, normalization, and filtering of signals.
- **Feature Engineering**: Extracting statistical, frequency-domain, and wavelet-transform-based features.
- **Machine Learning**: Ensemble learning combining Random Forest and Gradient Boosting regressors.
- **Hyperparameter Tuning**: Grid search for optimal model parameters.
- **Evaluation**: Mean Absolute Error (MAE) for systolic and diastolic predictions.

## Dataset

The project uses a dataset containing:
- **Signal Data**: ICA, CHROME, and POS outputs in sequence format.
- **Demographic Data**: Age and gender.
- **Target Variables**: Systolic (BP_Sys) and diastolic (BP_Dia) blood pressure.

### Preprocessing Steps:
1. **Gender Encoding**: Mapped 'F' to 0 and 'M' to 1.
2. **Sequence Padding**: Unified sequence lengths using TensorFlow's `pad_sequences`.
3. **Signal Filtering**: Applied median filtering, normalization, and baseline correction.
4. **Feature Engineering**:
   - Statistical features (mean, variance, skewness, kurtosis, etc.).
   - Frequency domain features using Welch's method.
   - Wavelet-transform-based features.

## Model Pipeline

1. **Train-Test Split**:
   - Features include engineered signal features and demographic data (age and gender).
   - Targets are BP_Sys and BP_Dia.
2. **Random Forest Regressor**:
   - Hyperparameters: `n_estimators`, `max_depth`.
   - Grid search for optimal tuning.
3. **Gradient Boosting Regressor**:
   - Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`.
   - Grid search for optimal tuning.
4. **Ensemble Prediction**:
   - Combined predictions of the two models for final output.
5. **Evaluation**:
   - Mean Absolute Error (MAE) computed for both systolic and diastolic predictions.

## Results

- **Systolic Prediction**:
  - Ensemble MAE: _Value_
- **Diastolic Prediction**:
  - Ensemble MAE: _Value_

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
