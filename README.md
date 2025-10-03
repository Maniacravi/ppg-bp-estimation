# 🩺 Cuffless Blood Pressure Estimation from PPG & ECG

End-to-end pipeline for **non-invasive, continuous blood pressure (BP) estimation** from photoplethysmography (**PPG**) and electrocardiogram (**ECG**) waveforms using **1D CNNs**. Includes data preprocessing, dataset creation (PPG-only and PPG+ECG), training, evaluation, and real-time inference benchmarking.

---

## 📦 Repository Structure

```
ppg-bp-estimation/
├── checkpoints/            # Saved model weights (best checkpoints)
├── dataset/                # Dataset + DataLoader utilities (PyTorch)
├── models/                 # Neural network architectures
├── notebooks/              # Exploration, preprocessing, training, evaluation
├── preprocessing/          # Filters, segmentation, preprocessing scripts
├── .gitignore
├── LICENSE                 # GPL-3.0
├── README.md
└── requirements.txt
```

> **Note:** Raw data and large `data/processed/*.npz` files are kept local (git-ignored). Prepare them via the preprocessing step below.

---

## 🧰 Environment

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📘 Dataset & Preprocessing

This project targets the **UCI Cuff-Less Blood Pressure Estimation dataset** (subset of MIMIC-II waveforms).

**Pipeline highlights (implemented in `preprocessing/`):**
- **Filtering**
  - PPG: **0.5–15 Hz** band-pass  
  - ECG: **0.5–40 Hz** band-pass  
  - ABP: **unfiltered** (preserve peak amplitudes)
- **SBP/DBP labeling**
  - SBP peaks via `scipy.signal.find_peaks` (≥0.2 s spacing, prominence ≥20)  
  - DBP as minima between consecutive SBP peaks
- **Segmentation**
  - **3-second** windows @125 Hz (375 samples), **75% overlap**
  - Labels per segment = **mean SBP/DBP within the window**
- **Outputs**
  - **PPG-only** dataset: shape `[N, L]`  
  - **PPG+ECG** dataset: shape `[N, 2, L]`  
  - Saved as compressed `.npz` under `data/processed/` (local)

Run preprocessing:

```bash
python preprocessing/preprocess_data.py
# Produces: data/processed/part_XXX_ppg_only.npz and part_XXX_ppg_ecg.npz
```

---

## 🧠 Model

A compact **1D CNN** for joint regression of **[SBP, DBP]** from short windows.

- Input:  
  - **PPG-only:** `[B, 1, 375]`  
  - **PPG+ECG:** `[B, 2, 375]`
- Core: stacked Conv1D + ReLU + pooling → Flatten → FC(128) → FC(2)
- Loss: **MSE**
- Optimizer: **Adam (lr=1e-3)**
- Training: **30 epochs**, batch size **256–4096**, **AMP** where available

---

## 🚀 Training & Evaluation

### Train (example, PPG-only)
Open the training notebook in `notebooks/` and point the dataloaders to your processed `.npz` paths, e.g.:

- `data/processed/ppg_clean_train.npz`  
- `data/processed/ppg_clean_val.npz`  
- `data/processed/ppg_clean_test.npz`

The loader wraps them as:
- `X`: `[N, L]` → tensor `[N, 1, L]`
- `y`: `[N, 2]` (columns: SBP, DBP)

**Logging:** tqdm progress bars, per-epoch Train/Val loss, RMSE per target, and best-model checkpointing under `checkpoints/`.

### Metrics reported
- **RMSE**, **MAE**, **R²**, **Pearson r**
- Bland–Altman statistics and **error histograms** (with ±1σ/±2σ/±3σ bands)

---

## 📈 Results (This Work)

### PPG-only (Test)
- **SBP RMSE:** **13.40 mmHg**  
- **DBP RMSE:** **8.28 mmHg**  
- **MAE:** SBP 10.80, DBP 6.39  
- **R² / r:** SBP 0.56 / 0.76, DBP 0.40 / 0.65

### PPG + ECG (Test)
- **SBP RMSE:** **10.56 mmHg**  
- **DBP RMSE:** **6.73 mmHg**  
- **MAE:** SBP 7.90, DBP 4.93  
- **R² / r:** SBP 0.76 / 0.87, DBP 0.65 / 0.81

**Interpretation:** Adding ECG provides consistent gains (leveraging timing/morphology related to pulse arrival/TT). These results are in line with cross-subject deep-learning performance reported on UCI/MIMIC-derived datasets.

---

## ⚡ Real-Time Inference

Measured per **3-second** window:

| Device | Avg time / window | Throughput |
|---|---:|---:|
| **GPU (CUDA)** | **~0.94 ms** | ~1067 samples/s |
| **CPU** | **~0.66 ms** | ~1520 samples/s |

This is **orders of magnitude faster than real-time**, enabling embedded / wearable deployment.

---

## 🧪 Academic Context (Selected)

| Study | Input | Model | SBP RMSE (mmHg) | DBP RMSE (mmHg) | Notes |
|------:|:------|:------|:----------------:|:----------------:|:------|
| Kachuee et al. (ISCAS 2015) | ECG + PPG | PTT-based | 11.17 | 5.36 | Classic PTT baseline |
| Liang et al. (Sensors 2018) | PPG | CNN | 13.1 | 7.7 | Morphology-based learning |
| Slapničar et al. (Sci Rep 2019) | ECG + PPG | DeepResNet | 9.4 | 6.5 | Subject-level training |
| **This work (2025)** | **PPG + ECG** | **1D CNN** | **10.6** | **6.7** | Comparable to recent deep-learning benchmarks |

---

## 📄 License

This repository is released under **GPL-3.0**.

---

## 📝 Citation

If you use this repository in academic work, please cite:

> Mani Ravi, *Cuffless Blood Pressure Estimation from PPG & ECG Signals using Deep Learning*, 2025.  
> GitHub: https://github.com/Maniacravi/ppg-bp-estimation

---

## 🙏 Acknowledgments

- UCI Cuff-Less BP Estimation dataset authors and maintainers.  
- Community work benchmarking UCI/MIMIC-derived datasets for cuffless BP.

---
