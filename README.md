# Knowledge-Infused Multi-Task Temporal Deep Learning for CTG Fetal Distress Prediction

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd CTG
```

### 2. Set Up the Environment
Create and activate a virtual environment, then install the dependencies:
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the Datasets
**Note:** Due to size constraints, datasets are not included in this repository.
1. Obtain the `cardiotocography.zip` and `ctu-chb-intrapartum-cardiotocography-database-1.0.0.zip` files.
2. Place these `.zip` files directly into the `data/raw/` directory.

### 4. Automated Preprocessing
The data engineering pipeline is fully complete and frozen (v1.0). To generate the PyTorch datasets:
```bash
python src/preprocessing/run_all.py
```
This script will:
- Extract the zip files.
- Parse the clinical metadata (pH values, etc.) from the WFDB headers.
- Filter, baseline-correct, and extract clinical features from the raw FHR/UC signals.
- Apply a 10-min stride (evaluation) and a balanced 2-min stride (training) for the distress cases.
- Save the final `train_dataset.pt`, `val_dataset.pt`, and `test_dataset.pt` into `data/processed/`.

To verify the clinical rules and feature extraction:
```bash
python src/preprocessing/consistency_audit.py
```

### 5. Model Building & Benchmarking (Current Phase)
We are currently in **Phase 3: Model Benchmarking**. 
- See `docs/model_evaluation_plan.md` for the benchmarking protocol.
- **IMPORTANT**: If you are an AI Agent or Developer contributing to a model, you **MUST** read and obey `AI_AGENT_RULES.md` before writing any code. All models must conform to the universal `(Batch, 2, 4800) -> (Batch, 128)` signature.
