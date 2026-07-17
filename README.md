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
**Note:** Datasets are NOT included in this Git repository due to size constraints. You must download them manually.

1. Obtain the `cardiotocography.zip` and `ctu-chb-intrapartum-cardiotocography-database-1.0.0.zip` files (Ask the team lead for the Google Drive link or download from the original sources).
2. Extract the datasets into the `data/raw/` directory so that your structure looks like this:
```text
data/
└── raw/
    ├── cardiotocography/
    └── ctu-chb-intrapartum/
```

### 4. Preprocessing
Once the datasets are in place, run the local preprocessing pipeline:
*(Implementation pending - see `docs/data_and_preprocessing_plan.md`)*
