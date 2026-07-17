import os
import zipfile
import wfdb
import pandas as pd
from pipeline import process_pipeline

def extract_datasets(base_dir: str):
    """
    Extracts the dataset zip files into the data/raw/ directory.
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
            # PhysioNet zips usually contain a parent folder, we'll extract directly
            zip_ref.extractall(raw_dir)
            # Rename the extracted physionet folder to our standard name
            extracted_name = os.path.join(raw_dir, "ctu-chb-intrapartum-cardiotocography-database-1.0.0")
            if os.path.exists(extracted_name):
                os.rename(extracted_name, ctu_dest)
    
    if os.path.exists(zip_uci) and not os.path.exists(uci_dest):
        print("Extracting UCI Cardiotocography dataset...")
        with zipfile.ZipFile(zip_uci, 'r') as zip_ref:
            zip_ref.extractall(uci_dest)
            
    return ctu_dest, uci_dest

def generate_metadata_from_headers(ctu_dir: str) -> str:
    """
    Scans all .hea files in the CTU-CHB directory, parses the clinical 
    comments (pH, Apgar, etc.), and generates a centralized CSV metadata file.
    """
    metadata_csv_path = os.path.join(ctu_dir, "clinical_metadata.csv")
    
    # If it already exists, return the path
    if os.path.exists(metadata_csv_path):
        print("Metadata CSV already exists.")
        return metadata_csv_path
        
    print("Generating clinical metadata CSV from WFDB headers...")
    records = []
    
    # Find all .hea files
    for file in os.listdir(ctu_dir):
        if file.endswith('.hea'):
            record_id = file.replace('.hea', '')
            record_path = os.path.join(ctu_dir, record_id)
            
            try:
                header = wfdb.rdheader(record_path)
                # Parse comments
                meta_dict = {'record_id': record_id}
                for comment in header.comments:
                    parts = comment.rsplit(maxsplit=1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        val = parts[1].strip()
                        meta_dict[key] = val
                        
                # Ensure we have a pH value to use for the target
                if 'ph' in meta_dict:
                    try:
                        meta_dict['ph'] = float(meta_dict['ph'])
                        records.append(meta_dict)
                    except ValueError:
                        pass # Skip if pH is not a float
            except Exception as e:
                print(f"Failed to read header {record_id}: {e}")
                
    df = pd.DataFrame(records)
    if len(df) > 0:
        df.to_csv(metadata_csv_path, index=False)
        print(f"Successfully generated metadata for {len(df)} records.")
    else:
        print("Warning: No records found with valid pH values.")
        
    return metadata_csv_path

if __name__ == "__main__":
    # Base directory of the project
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    print("=== Step 1: Data Extraction ===")
    ctu_dest, uci_dest = extract_datasets(BASE_DIR)
    
    print("\n=== Step 2: Metadata Aggregation ===")
    metadata_path = generate_metadata_from_headers(ctu_dest)
    
    print("\n=== Step 3: Main Orchestration Pipeline ===")
    out_dir = os.path.join(BASE_DIR, "data", "processed")
    process_pipeline(ctu_dest, metadata_path, out_dir)
    
    print("\nAll preprocessing complete!")
