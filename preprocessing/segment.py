"""
segment.py — segmentation utilities for BP estimation

Splits signals into overlapping windows and computes SBP/DBP per segment.
"""

import numpy as np


def segment_ppg_signal(filtered_ppg, abp_signal, systolic_peaks, diastolic_peaks,
                       fs, window_sec=3.0, overlap_ratio=0.75):
    """
    Segment PPG into overlapping windows and assign SBP/DBP labels per segment.

    Args:
        filtered_ppg (np.ndarray): filtered PPG signal
        abp_signal (np.ndarray): unfiltered ABP signal (for SBP/DBP extraction)
        systolic_peaks (np.ndarray): indices of systolic peaks in ABP
        diastolic_peaks (np.ndarray): indices of diastolic peaks in ABP
        fs (int): sampling frequency (Hz)
        window_sec (float): segment duration in seconds (default 3)
        overlap_ratio (float): overlap fraction between 0 and 1 (default 0.75)

    Returns:
        segments_ppg (np.ndarray): (num_segments, window_length)
        segments_sbp (np.ndarray): SBP per segment
        segments_dbp (np.ndarray): DBP per segment
    """
    segment_length = int(window_sec * fs)
    overlap = int(overlap_ratio * segment_length)
    step_size = segment_length - overlap
    num_segments = (len(filtered_ppg) - overlap) // step_size

    segments_ppg, segments_sbp, segments_dbp = [], [], []

    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_length
        if end_idx > len(filtered_ppg):
            break

        segment_ppg = filtered_ppg[start_idx:end_idx]

        # Find systolic/diastolic peaks that fall within this window
        systolic_in_seg = systolic_peaks[(systolic_peaks >= start_idx) & (systolic_peaks < end_idx)]
        diastolic_in_seg = diastolic_peaks[(diastolic_peaks >= start_idx) & (diastolic_peaks < end_idx)]

        # Label = mean SBP / DBP in that window
        if len(systolic_in_seg) > 0:
            sbp = np.mean(abp_signal[systolic_in_seg])
        else:
            sbp = np.nan

        if len(diastolic_in_seg) > 0:
            dbp = np.mean(abp_signal[diastolic_in_seg])
        else:
            dbp = np.nan

        segments_ppg.append(segment_ppg)
        segments_sbp.append(sbp)
        segments_dbp.append(dbp)

    return np.array(segments_ppg), np.array(segments_sbp), np.array(segments_dbp)
