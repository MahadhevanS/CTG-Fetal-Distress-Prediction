import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ingestion import load_ctu_chb_record, load_clinical_metadata
from signal_quality import get_valid_windows
from filtering import interpolate_missing, apply_lowpass_filter
from baseline import calculate_iterative_baseline
from features import calculate_variability, detect_accelerations, detect_decelerations
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knowledge.figo import classify_figo

def process_pipeline(raw_data_dir: str, metadata_path: str, output_dir: str):
    """
    Master orchestration script for CTG preprocessing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading metadata...")
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file {metadata_path} not found. Skipping pipeline for now.")
        return
        
    metadata = load_clinical_metadata(metadata_path)
    # Assume metadata has columns 'record_id' and 'ph'
    metadata = metadata.set_index('record_id') if 'record_id' in metadata.columns else metadata
    
    # Stratified Patient-Level Split
    # We split IDs to prevent data leakage
    records = list(metadata.index)
    # Define primary outcome: Distress = 1 if pH <= 7.15 else 0
    y_patient = (metadata['ph'] <= 7.15).astype(int).values
    
    train_ids, test_val_ids, y_train, y_test_val = train_test_split(
        records, y_patient, test_size=0.3, stratify=y_patient, random_state=42
    )
    val_ids, test_ids, _, _ = train_test_split(
        test_val_ids, y_test_val, test_size=0.5, stratify=y_test_val, random_state=42
    )
    
    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    
    window_samples = int(20 * 60 * 4) # 20 minutes at 4Hz
    
    for split_name, ids in splits.items():
        print(f"Processing {split_name} split ({len(ids)} patients)...")
        X_data = []
        Y_primary = []
        Y_figo = []
        Y_features = []
        Record_ids = []
        
        for record_id in ids:
            record_path = os.path.join(raw_data_dir, str(record_id))
            if not os.path.exists(record_path + '.dat'):
                continue
                
            fhr, uc, fs = load_ctu_chb_record(record_path)
            if len(fhr) == 0:
                continue
                
            is_distress = (metadata.loc[record_id, 'ph'] <= 7.15)
            
            # Imbalance Handling (Training Set ONLY)
            # Normal: ~4 windows per patient. Distress: ~20 windows (stride=2min) 
            # This achieves approx 1200 Normal : 1400 Distress for 50:50 balance
            if split_name == 'train':
                stride = int(2 * 60 * fs) if is_distress else int(10 * 60 * fs)
            else:
                stride = int(10 * 60 * fs) # Fixed evaluation stride
                
            # Only take windows from the last hour (clinically relevant horizon)
            last_hour_samples = int(60 * 60 * fs)
            if len(fhr) > last_hour_samples:
                fhr = fhr[-last_hour_samples:]
                uc = uc[-last_hour_samples:]
                
            valid_starts = get_valid_windows(fhr, window_samples, stride)
            
            for start in valid_starts:
                fhr_win = fhr[start : start + window_samples]
                uc_win = uc[start : start + window_samples]
                
                # Signal Processing
                fhr_win = interpolate_missing(fhr_win)
                fhr_win = apply_lowpass_filter(fhr_win, fs)
                uc_win = apply_lowpass_filter(uc_win, fs)
                
                # Baseline & Normalization
                baseline = calculate_iterative_baseline(fhr_win)
                fhr_norm = fhr_win - baseline # Baseline Corrected FHR
                
                # Feature Extraction
                stv, ltv = calculate_variability(fhr_win, fs)
                accels = detect_accelerations(fhr_win, baseline, fs)
                decels = detect_decelerations(fhr_win, baseline, uc_win, fs)
                
                # Pseudo-Labels (FIGO)
                base_val = np.mean(baseline)
                figo_class = classify_figo(base_val, ltv, accels, decels)
                
                # Collate Data
                # Stack FHR and UC as channels: shape (2, window_samples)
                window_tensor = np.vstack((fhr_norm, uc_win))
                
                features_target = [
                    base_val, stv, ltv, accels, 
                    decels['early'], decels['late'], decels['variable'], decels['prolonged']
                ]
                
                X_data.append(window_tensor)
                Y_primary.append(int(is_distress))
                Y_figo.append(int(figo_class))
                Y_features.append(features_target)
                Record_ids.append((record_id, start, start + window_samples))
                
        if len(X_data) > 0:
            # Convert to PyTorch tensors and save
            torch.save({
                'X': torch.tensor(np.array(X_data), dtype=torch.float32),
                'y_primary': torch.tensor(np.array(Y_primary), dtype=torch.long),
                'y_figo': torch.tensor(np.array(Y_figo), dtype=torch.long),
                'y_features': torch.tensor(np.array(Y_features), dtype=torch.float32),
                'metadata': Record_ids
            }, os.path.join(output_dir, f'{split_name}_dataset.pt'))
            print(f"Saved {split_name}_dataset.pt with {len(X_data)} windows.")
            
if __name__ == "__main__":
    # Define paths based on project structure
    RAW_DIR = "../../data/raw/ctu-chb-intrapartum/"
    METADATA_FILE = "../../data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
    OUT_DIR = "../../data/processed/"
    
    # process_pipeline(RAW_DIR, METADATA_FILE, OUT_DIR)
    print("Pipeline script initialized. Waiting for dataset extraction to run.")
