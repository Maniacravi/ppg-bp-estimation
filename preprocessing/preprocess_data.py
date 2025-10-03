"""
Preprocess all raw PPG-ABP-ECG data for BP estimation.

Creates two datasets per file:
  1. PPG-only dataset  (X shape: [num_segments, window_len])
  2. PPG+ECG dataset   (X shape: [num_segments, 2, window_len])
Labels are [SBP, DBP] per segment.
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
from filters import bandpass_filter
from segment import segment_ppg_signal


# -------------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------------
def load_mat_data(mat_path):
    """Load .mat data and extract PPG, ABP, ECG signals."""
    data = loadmat(mat_path)
    p = data.get('p', None)
    if p is None:
        raise ValueError(f"Could not find key 'p' in {mat_path}")

    num_records = len(p[0])
    print(f"📦 Found {num_records} records in {os.path.basename(mat_path)}")

    ppg_list, abp_list, ecg_list = [], [], []
    for i in range(num_records):
        try:
            ppg = p[0][i][0]
            abp = p[0][i][1]
            ecg = p[0][i][2]
            ppg_list.append(ppg.squeeze())
            abp_list.append(abp.squeeze())
            ecg_list.append(ecg.squeeze())
        except Exception as e:
            print(f"⚠️ Skipping record {i}: {e}")
            continue

    return ppg_list, abp_list, ecg_list


# -------------------------------------------------------------------------
# 2. ABP peak detection for SBP / DBP
# -------------------------------------------------------------------------
def detect_peaks(abp_signal, fs=125):
    """Detect systolic (maxima) and diastolic (minima) peaks from ABP."""
    # Detect systolic peaks using find_peaks. 
    systolic_peaks, _ = find_peaks(abp_signal, distance=fs*0.2, prominence=20)  # Minimum distance of 0.2 seconds between peaks
    # Detect diastolic peaks as the minimum point between each pair of systolic peaks
    diastolic_peaks = []
    for i in range(len(systolic_peaks) - 1):
        start = systolic_peaks[i]
        end = systolic_peaks[i + 1]
        if end > start:  # Ensure valid range
            diastolic_peak = start + np.argmin(abp_signal[start:end])
            diastolic_peaks.append(diastolic_peak)
    systolic_peaks = np.array(systolic_peaks)
    diastolic_peaks = np.array(diastolic_peaks)
    return systolic_peaks, diastolic_peaks


# -------------------------------------------------------------------------
# 3. Preprocessing pipeline
# -------------------------------------------------------------------------
def preprocess_all(input_dir="./data", output_dir="./data/processed",
                   window_sec=3.0, overlap_ratio=0.75, fs=125):
    """Preprocess all .mat files to create both PPG-only and PPG+ECG datasets."""
    os.makedirs(output_dir, exist_ok=True)
    mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mat")])

    for file in mat_files:
        mat_path = os.path.join(input_dir, file)
        print(f"\n🚀 Processing file: {file}")

        ppg_list, abp_list, ecg_list = load_mat_data(mat_path)

        all_ppg_segments, all_ppg_ecg_segments, all_labels = [], [], []

        for idx, (ppg_signal, abp_signal, ecg_signal) in enumerate(zip(ppg_list, abp_list, ecg_list)):
            # ---- Filter signals ----
            filtered_ppg = bandpass_filter(ppg_signal, fs=fs, lowcut=0.5, highcut=15)
            filtered_ecg = bandpass_filter(ecg_signal, fs=fs, lowcut=0.5, highcut=40)
            filtered_abp = abp_signal  # do not filter ABP (preserve peak amplitudes)

            # ---- Detect SBP/DBP ----
            systolic_peaks, diastolic_peaks = detect_peaks(filtered_abp, fs)

            # ---- Segment signals ----
            segments_ppg, segments_sbp, segments_dbp = segment_ppg_signal(
                filtered_ppg, filtered_abp, systolic_peaks, diastolic_peaks,
                fs=fs, window_sec=window_sec, overlap_ratio=overlap_ratio
            )

            # ---- Segment ECG with same windowing ----
            segment_len = segments_ppg.shape[1]
            step_size = int(segment_len * (1 - overlap_ratio))
            num_segments = len(segments_ppg)
            segments_ecg = np.zeros_like(segments_ppg)

            for i in range(num_segments):
                start_idx = i * step_size
                end_idx = start_idx + segment_len
                if end_idx <= len(filtered_ecg):
                    segments_ecg[i, :] = filtered_ecg[start_idx:end_idx]

            # ---- Clean invalid labels ----
            valid_mask = ~np.isnan(segments_sbp) & ~np.isnan(segments_dbp)
            if np.sum(valid_mask) == 0:
                continue

            segments_ppg = segments_ppg[valid_mask]
            segments_ecg = segments_ecg[valid_mask]
            labels = np.stack([segments_sbp[valid_mask], segments_dbp[valid_mask]], axis=1)

            # ---- Store both dataset versions ----
            X_ppg_only = segments_ppg
            X_ppg_ecg = np.stack([segments_ppg, segments_ecg], axis=1)  # [N, 2, L]

            all_ppg_segments.append(X_ppg_only)
            all_ppg_ecg_segments.append(X_ppg_ecg)
            all_labels.append(labels)

        # ---- Combine and save ----
        if not all_ppg_segments:
            print(f"⚠️ No valid segments found for {file}")
            continue

        all_ppg_segments = np.concatenate(all_ppg_segments, axis=0)
        all_ppg_ecg_segments = np.concatenate(all_ppg_ecg_segments, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        base_name = file.replace(".mat", "")
        save_path_ppg = os.path.join(output_dir, f"{base_name}_ppg_only.npz")
        save_path_ppg_ecg = os.path.join(output_dir, f"{base_name}_ppg_ecg.npz")

        np.savez_compressed(save_path_ppg, X=all_ppg_segments, y=all_labels)
        np.savez_compressed(save_path_ppg_ecg, X=all_ppg_ecg_segments, y=all_labels)

        print(f"✅ Saved {len(all_ppg_segments)} segments → {save_path_ppg}")
        print(f"✅ Saved {len(all_ppg_ecg_segments)} segments → {save_path_ppg_ecg}")


# -------------------------------------------------------------------------
# 4. Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    preprocess_all()
