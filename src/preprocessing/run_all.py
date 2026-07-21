import os
import zipfile
import wfdb
import pandas as pd
from pipeline import process_pipeline
from uci_pipeline import preprocess_uci   # GAP 3 FIX


def extract_datasets(base_dir: str):
    """
    Extracts the dataset zip files into the data/raw/ directory.
    Skips extraction if destination already exists.

    Args:
        base_dir (str): Project root directory.

    Returns:
        Tuple[str, str]: (ctu_dest, uci_dest) — extracted dataset paths.
    """
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_ctu = os.path.join(raw_dir, "ctu-chb-intrapartum-cardiotocography-database-1.0.0.zip")
    zip_uci = os.path.join(raw_dir, "cardiotocography.zip")

    ctu_dest = os.path.join(raw_dir, "ctu-chb-intrapartum")
    uci_dest = os.path.join(raw_dir, "cardiotocography")

    if os.path.exists(zip_ctu) and not os.path.exists(ctu_dest):
        print("Extracting CTU-CHB Intrapartum dataset...")
        with zipfile.ZipFile(zip_ctu, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
            extracted_name = os.path.join(
                raw_dir,
                "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
            )
            if os.path.exists(extracted_name):
                os.rename(extracted_name, ctu_dest)
    elif os.path.exists(ctu_dest):
        print("CTU-CHB Intrapartum dataset already extracted.")
    else:
        print("[WARNING] CTU-CHB zip not found. Skipping extraction.")

    if os.path.exists(zip_uci) and not os.path.exists(uci_dest):
        print("Extracting UCI Cardiotocography dataset...")
        with zipfile.ZipFile(zip_uci, 'r') as zip_ref:
            zip_ref.extractall(uci_dest)
    elif os.path.exists(uci_dest):
        print("UCI Cardiotocography dataset already extracted.")
    else:
        print("[WARNING] UCI CTG zip not found. Skipping extraction.")

    return ctu_dest, uci_dest


def generate_metadata_from_headers(ctu_dir: str) -> str:
    """
    Scans all .hea files in the CTU-CHB directory, parses the clinical
    comments (pH, Apgar, etc.), and generates a centralised CSV metadata file.

    Skips generation if the metadata CSV already exists (idempotent).

    Args:
        ctu_dir (str): Path to the extracted CTU-CHB dataset directory.

    Returns:
        str: Path to the generated (or existing) clinical_metadata.csv.
    """
    metadata_csv_path = os.path.join(ctu_dir, "clinical_metadata.csv")

    if os.path.exists(metadata_csv_path):
        print("Metadata CSV already exists. Skipping generation.")
        return metadata_csv_path

    print("Generating clinical metadata CSV from WFDB headers...")
    records = []

    for file in os.listdir(ctu_dir):
        if not file.endswith('.hea'):
            continue

        record_id   = file.replace('.hea', '')
        record_path = os.path.join(ctu_dir, record_id)

        try:
            header   = wfdb.rdheader(record_path)
            meta_dict = {'record_id': record_id}

            for comment in header.comments:
                parts = comment.rsplit(maxsplit=1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    val = parts[1].strip()
                    meta_dict[key] = val

            # Only include records with a valid numeric pH value
            if 'ph' in meta_dict:
                try:
                    meta_dict['ph'] = float(meta_dict['ph'])
                    records.append(meta_dict)
                except ValueError:
                    pass  # Skip records with non-numeric pH

        except Exception as e:
            print(f"  [WARNING] Failed to read header {record_id}: {e}")

    df = pd.DataFrame(records)
    if len(df) > 0:
        df.to_csv(metadata_csv_path, index=False)
        print(f"Successfully generated metadata for {len(df)} records.")
    else:
        print("[ERROR] No records found with valid pH values.")

    return metadata_csv_path


if __name__ == "__main__":
    # Resolve project base directory relative to this script's location
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # -------------------------------------------------------------------
    # Step 1: Data Extraction
    # -------------------------------------------------------------------
    print("=" * 50)
    print("  Step 1: Data Extraction")
    print("=" * 50)
    ctu_dest, uci_dest = extract_datasets(BASE_DIR)

    # -------------------------------------------------------------------
    # Step 2: Metadata Aggregation (CTU-CHB WFDB headers → CSV)
    # -------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  Step 2: Metadata Aggregation")
    print("=" * 50)
    metadata_path = generate_metadata_from_headers(ctu_dest)

    # -------------------------------------------------------------------
    # Step 3: CTU-CHB Main Orchestration Pipeline
    # Includes: spike removal (G1), prediction horizon (G2),
    #           fs validation (G4), Z-score normalisation (G5)
    # -------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  Step 3: CTU-CHB Main Orchestration Pipeline")
    print("=" * 50)
    out_dir = os.path.join(BASE_DIR, "data", "processed")
    process_pipeline(ctu_dest, metadata_path, out_dir)

    # -------------------------------------------------------------------
    # Step 4 (GAP 3 FIX): UCI SisPorto Tabular Preprocessing
    # -------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  Step 4: UCI SisPorto Tabular Preprocessing")
    print("=" * 50)
    if os.path.exists(uci_dest):
        preprocess_uci(uci_dest, out_dir)
    else:
        print(f"  [SKIP] UCI dataset directory not found: {uci_dest}")
        print("  Ensure the UCI CTG zip file is in data/raw/ and re-run.")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  All preprocessing steps complete!")
    print("=" * 50)
    print("\nOutputs written to:", out_dir)
    print("  CTU-CHB splits : train_dataset.pt, val_dataset.pt, test_dataset.pt")
    print("  CTU-CHB scaler : ctu_signal_scaler.npz")
    print("  UCI splits     : uci_train_dataset.pt, uci_val_dataset.pt, uci_test_dataset.pt")
    print("  UCI scaler     : uci_scaler.joblib")
    print("\nNext step: Run consistency_audit.py to validate all outputs.")
