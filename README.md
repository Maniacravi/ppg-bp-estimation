# ppg-bp-estimation
Cuffless Blood Pressure Estimation from PPG signals using machine learning and deep learning models.

## Overview
This project explores whether photoplethysmography (PPG) signals can be used to estimate systolic and diastolic blood pressure without a cuff. 
It demonstrates signal preprocessing, feature extraction, and the use of deep learning models (CNNs/LSTMs) for regression tasks on biomedical signals.

The goal: build models that can predict blood pressure values from PPG waveforms — an important step toward wearable and continuous health monitoring.

## Data
The data used in this repo is from UCI's Machine Learning Repository, obtained through [Kaggle](https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset/data). 

**References:**
- M. Kachuee, M. M. Kiani, H. Mohammadzade, M. Shabany, *Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time*, IEEE ISCAS, 2015.  
- M. Kachuee, M. M. Kiani, H. Mohammadzadeh, M. Shabany, *Cuff-Less Blood Pressure Estimation Algorithms for Continuous Health-Care Monitoring*, IEEE TBME, 2016.

## Approach
1. **Preprocessing**  
   - Bandpass filtering  
   - Windowing into fixed-length segments  
   - Normalization of signals  

2. **Baseline Models**  
   - Linear regression and random forest on engineered features  

3. **Deep Learning Models**  
   - 1D CNN for waveform feature extraction  
   - LSTM/GRU for temporal sequence modeling  
   - Hybrid CNN-LSTM  

4. **Evaluation Metrics**  
   - Mean Absolute Error (MAE)  
   - Root Mean Square Error (RMSE)  
   - Bland–Altman plots for clinical agreement  

## Results
- Baseline models vs. deep learning models compared  
- CNN-LSTM hybrid achieved lowest MAE on test set  
- Bland–Altman plots show good agreement with reference BP  

## How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/Maniacravi/ppg-bp-estimation.git
   cd ppg-bp-estimation
