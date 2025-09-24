# MIMIC Preprocessor

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A streamlined, efficient, and configurable pipeline for processing the raw MIMIC-III and MIMIC-IV datasets into analysis-ready formats. This toolkit handles the complexities of cleaning, merging, and feature engineering, allowing researchers to focus on data analysis and model development.

## Features

- **Dual Dataset Support**: Process both MIMIC-III (v1.4) and MIMIC-IV (v3.1, note v2.2) datasets.
- **Modular Processing**: Choose which parts of the MIMIC-IV dataset to process (`ehr`, `note`, `icd`) and whether to merge them.
- **Clean Outputs**: Generates cleaned, analysis-ready data in the efficient Parquet file format.

## Project Structure

```bash
├── main.py                 # Main entry point for the CLI
├── src/                    # Source code for the project
│   ├── __init__.py
│   ├── mimic_iii_processor.py
│   ├── mimic_iv_processor.py
│   └── utils.py
├── mimic_datasets/         # Directory for raw and processed data (not versioned)
├── pyproject.toml          # Project metadata and dependencies for uv/pip
├── README.md               # This file
└── uv.lock                 # Lockfile for reproducible dependencies
```

## Prerequisites

Follow these steps to set up the project environment and download the necessary datasets.

### 1. Clone the Repository

```bash
git clone https://github.com/PKU-AICare/mimic_preprocessor.git
cd mimic_preprocessor
```

### 2. Set Up the Python Environment (using uv)

This project uses [**`uv`**](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

First, install `uv`:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Next, create a virtual environment and install the dependencies defined in `pyproject.toml`:

```bash
# Install dependencies using the uv.lock file
uv sync

# Activate the environment
# On macOS and Linux
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
```

### 3. Download MIMIC Datasets

Access to MIMIC datasets is restricted and requires credentialing on PhysioNet.

1. **Gain Access**: Complete the required training and apply for access on the official PhysioNet website:

   - [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)
   - [MIMIC-IV Clinical Database v3.1](https://physionet.org/content/mimiciv/3.1/)
   - [MIMIC-IV-Note Database v2.2](https://physionet.org/content/mimic-iv-note/2.2/)

2. **Organize Files**: After downloading, unzip the files and place them into the `mimic_datasets` directory following the exact structure below. The processing scripts rely on these default paths.

```bash
  mimic-processor/
  └── mimic_datasets/
      ├── mimic_iii/
      │   └── 1.4/
      │       └── raw/
      │           ├── ADMISSIONS.csv
      │           ├── PATIENTS.csv
      │           └── NOTEEVENTS.csv
      │           └── ... (all other MIMIC-III files)
      ├── mimic_iv/
      │   └── 3.1/
      │       ├── hosp/
      │       │   ├── patients.csv
      │       │   └── ...
      │       ├── icu/
      │       │   ├── chartevents.csv
      │       │   └── ...
      │       ├── config.toml         # Config file for MIMIC-IV
      │       └── mimic4_item2var.csv # Variable mapping file for MIMIC-IV
      └── mimic_iv_note/
          └── 2.2/
              └── note/
                  ├── discharge.csv
                  └── ...
```

## Usage

All processing is handled via the `main.py` script. You can specify the dataset (`mimic3` or `mimic4`) and other options.

### General Command Structure

```bash
python main.py <dataset> [options]
```

### Processing MIMIC-III

This command runs the complete pipeline for MIMIC-III, which processes and merges patient demographics with clinical notes.

```bash
python main.py mimic3
```

- **Logs**: Will be printed to the console by default.
- **Output**: A single file named `mimic_iii_note_label.parquet` will be saved in `mimic_datasets/mimic_iii/1.4/processed/`.

### Processing MIMIC-IV

MIMIC-IV processing is more modular. You can specify which parts to process and whether to merge them.

#### Example 1: Full Pipeline (Recommended)

Process EHR, notes, merge them, and then process ICD/prescription data.

```bash
python main.py mimic4 --parts ehr note icd --merge
```

#### Example 2: Process EHR and Notes Separately

If you only want the processed EHR and note files without merging them.

```bash
python main.py mimic4 --parts ehr note
```

#### Example 3: Process ICD Data Only

This assumes you have already run the `--merge` step, as it depends on the merged output file.

```bash
python main.py mimic4 --parts icd
```

### Advanced Options

- **Log to a File**: Use the `--log_file` argument to redirect all logs to a file.

  ```bash
  python main.py mimic4 --merge --log_file processing.log
  ```

- **Custom Data Paths**: If your data is not in the default location, specify the paths:

  ```bash
  python main.py mimic4 --merge \
    --mimic4_data_dir /path/to/mimic4_data \
    --mimic4_note_dir /path/to/mimic4_notes \
    --mimic4_processed_dir /path/to/your/output
  ```

## Data Processing Logic and Outputs

### MIMIC-III Processor (`src/mimic_iii_processor.py`)

The MIMIC-III pipeline integrates patient demographics, admission details, and clinical notes into a single file.

1. **Notes Processing**:
   - Loads `NOTEEVENTS.csv`.
   - Removes duplicates and notes flagged as errors.
   - Filters to keep only 'Nursing', 'Nursing/other', and 'Physician' notes.
   - Groups notes by `PatientID`, `AdmissionID`, and `RecordDate`, concatenating text from different categories for the same day.
   - Selects the first note entry per admission.
   - Applies text preprocessing: lowercasing, removing special characters, and standardizing whitespace.

2. **Demographics & Admissions Processing**:
   - Loads and merges `ADMISSIONS.csv` and `PATIENTS.csv`.
   - Calculates patient `Age` at the time of admission. Ages >= 90 or <= 0 are capped at 90.
   - Computes `InHospitalOutcome` (mortality) based on patient's date of death relative to admission and discharge times.
   - Formats `Gender` into a binary value (1 for Male, 0 for Female).

#### Final Output: `mimic_iii_note_label.parquet`

A single file containing one record per admission, with key columns:

- `RecordID`: Unique identifier (`PatientID` + `AdmissionID`).
- `PatientID`, `AdmissionID`.
- `Age`, `Gender`, `Race`.
- `InHospitalOutcome`: Binary mortality label.
- `Text`: The fully cleaned and concatenated clinical note text for that admission.

### MIMIC-IV Processor (`src/mimic_iv_processor.py`)

The MIMIC-IV pipeline is divided into distinct, controllable modules.

#### 1. EHR Processing (`--parts ehr`)

This module processes patient stays and clinical events.

- **Stays Module**:
  - Merges `patients.csv`, `admissions.csv`, and `icustays.csv`.
  - Calculates `InUnitOutcome` (mortality in ICU), `InHospitalOutcome` (mortality in hospital), and `Outcome` (mortality in hospital or ICU).
  - Calculates `Readmission` (30-day readmission) using an efficient, vectorized approach.
  - **Output**: `mimic4_formatted_patients.parquet`.

- **Events Module**:
  - Processes the massive `chartevents.csv` in memory-efficient chunks.
  - Maps `itemid` to clinical variables using `mimic4_item2var.csv`.
  - Formats values (e.g., converting temperature from Fahrenheit to Celsius).
  - Pivots the data from long to wide format, where each row is a timestamp and columns are clinical variables.
  - **Output**: `mimic4_formatted_events.parquet`.

- **Final EHR Merge**:
  - The stays and events tables are merged on `PatientID`, `AdmissionID`, and `StayID`.
  - **Output**: `mimic4_formatted_ehr.parquet`.

#### 2. Note Processing (`--parts note`)

- Processes `discharge.csv` from the MIMIC-IV-Note dataset.
- Applies the same text preprocessing logic as in MIMIC-III (lowercasing, cleaning, etc.).
- **Output**: `mimic4_discharge_note.parquet`.

#### 3. EHR & Note Merging (`--merge`)

- This step combines the outputs from the EHR and Note processing modules.
- **7-Day Window Logic**: Before merging, it aggregates clinical events within a 7-day sliding window for each admission to reduce data granularity and align time-series data with static notes.
- **Output**: `mimic4_discharge_note_ehr.parquet`. This is the primary file for multimodal analysis.

#### 4. ICD & Prescription Processing (`--parts icd`)

- This module depends on the `mimic4_discharge_note_ehr.parquet` file created by the `--merge` step.
- It loads `diagnoses_icd.csv`, `procedures_icd.csv`, and `prescriptions.csv`.
- It filters these files to retain only the records corresponding to the patient admissions present in the main merged cohort.
- **Outputs**:
  - `mimic4_diagnoses_icd.parquet`
  - `mimic4_procedures_icd.parquet`
  - `mimic4_prescriptions.parquet`

## Data Field Descriptions

### Basic Patient Information

- `RecordID`: Unique identifier for the record, obtained by concatenating `PatientID` and `AdmissionID`.
- `PatientID`: Unique identifier for the patient.
- `AdmissionID`: Unique identifier for the admission.
- `StayID`: Unique identifier for the stay.
- `RecordTime`: Timestamp of the record.
- `NoteRecordTime`: Timestamp of the note record.
- `Age`: Age of the patient.
- `Sex`: Sex of the patient.

### Prediction Labels

- `InUnitOutcome`: Mortality in ICU.
- `InHospitalOutcome`: Mortality in hospital.
- `Outcome`: Mortality in hospital or ICU.
- `Readmission`: 30-day readmission after discharge.

### Predictive Features

- 17 labtest features, including 5 categorical features and 12 numerical features.
- `Text`: Discharge note text.
