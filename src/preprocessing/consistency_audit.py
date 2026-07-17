import os
import torch
import numpy as np

def run_consistency_audit(processed_dir: str):
    """
    Audits the generated PyTorch datasets for clinical consistency, 
    class distribution, and dataset statistics.
    """
    splits = ['train', 'val', 'test']
    
    print("=========================================")
    print("       DATASET CONSISTENCY AUDIT         ")
    print("=========================================\n")
    
    for split in splits:
        file_path = os.path.join(processed_dir, f'{split}_dataset.pt')
        if not os.path.exists(file_path):
            continue
            
        data = torch.load(file_path)
        X = data['X']
        y_primary = data['y_primary'].numpy()
        figo = data['y_figo'].numpy()
        y_features = data['y_features'].numpy()
        
        n_windows = len(X)
        distress_count = (y_primary == 1).sum()
        normal_count = (y_primary == 0).sum()
        
        f_0 = (figo == 0).sum()
        f_1 = (figo == 1).sum()
        f_2 = (figo == 2).sum()
        
        print(f"[{split.upper()} SPLIT]")
        print(f"Total Windows : {n_windows}")
        print(f"pH Outcome    : {normal_count} Normal | {distress_count} Distress")
        print(f"FIGO Classes  : {f_0} Normal | {f_1} Suspicious | {f_2} Pathological")
        
        # Clinical consistency check:
        # If FIGO == Pathological (2), we expect either low baseline, late decels, 
        # prolonged decels, or extreme variability issues.
        pathological_idx = np.where(figo == 2)[0]
        if len(pathological_idx) > 0:
            late_decels = y_features[pathological_idx, 5]
            prol_decels = y_features[pathological_idx, 7]
            baselines = y_features[pathological_idx, 0]
            
            has_late = (late_decels > 0).sum()
            has_prol = (prol_decels > 0).sum()
            low_base = (baselines < 100).sum()
            
            explained = has_late + has_prol + low_base
            print(f"Pathological consistency: {explained}/{len(pathological_idx)} windows have explicit late/prolonged decels or <100 baseline.")
            
        print("\n--- FIGO vs pH Correlation ---")
        for f_class, f_name in [(0, "Normal    "), (1, "Suspicious"), (2, "Patholog. ")]:
            idx = (figo == f_class)
            ph_norm = (y_primary[idx] == 0).sum()
            ph_dist = (y_primary[idx] == 1).sum()
            print(f"FIGO {f_name} -> pH Normal: {ph_norm:4d} | pH Distress: {ph_dist:4d}")
            
        print("-" * 40)

if __name__ == "__main__":
    OUT_DIR = "../../data/processed/"
    if os.path.exists(OUT_DIR):
        run_consistency_audit(OUT_DIR)
    else:
        # Fallback to absolute if run from wrong directory
        run_consistency_audit(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")))
