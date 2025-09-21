# MIMIC Preprocessor

This is an open-source tool for processing the MIMIC-III and MIMIC-IV (Medical Information Mart for Intensive Care) datasets. The tool is designed to process and transform MIMIC's Electronic Health Records (EHR) and clinical notes data, making it more suitable for machine learning research.

## Dataset Source

- MIMIC-III v1.4: <https://physionet.org/content/mimiciii/1.4/>
- MIMIC-IV EHR data: MIMIC-IV v3.1 (<https://physionet.org/content/mimiciv/3.1/>)
- MIMIC-IV Clinical notes data: MIMIC-IV v2.2 (<https://physionet.org/content/mimic-iv-note/2.2/>)

## Features

- MIMIC-III Clinical Notes Processing
  - Clinical note text preprocessing and cleaning
  - Note categorization (Physician, Nursing, Nursing/other)
  - Text data standardization and de-identification
  - Patient admission information processing
  - Mortality prediction label generation
  - Integration of clinical notes with patient data

- MIMIC-IV EHR Data Processing
  - Patient basic information processing
  - Hospital admission record processing
  - ICU stay record processing
  - Clinical event data processing
  - Mortality prediction label generation
  - Readmission prediction label generation

- MIMIC-IV Clinical Notes Processing
  - Discharge note text preprocessing
  - Text data cleaning and standardization
  - Integration with EHR data

## Data Output

Processed data will be saved in Parquet format, including the following files:

### MIMIC-III Data

- `mimic_iii_note.parquet`: Processed clinical notes data
- `mimic_iii_patients.parquet`: Processed patient information with mortality labels
- `mimic_iii_note_label.parquet`: Complete dataset integrating clinical notes with patient data

### MIMIC-IV Data

- `mimic4_formatted_icustays.parquet`: Processed ICU stay records
- `mimic4_formatted_events.parquet`: Processed clinical event data
- `mimic4_discharge_note.parquet`: Processed discharge notes
- `mimic4_discharge_note_ehr.parquet`: Complete dataset integrating EHR and clinical notes

## Requirements

- Python 3.11+
- pandas == 2.2.3
- numpy == 2.2.3
- tomli == 2.2.1

## Usage

1. Install Dependencies

   ```bash
   pip install pandas numpy tomli
   ```

2. Prepare Data
   - Sign up for MIMIC-III and MIMIC-IV access, download the data and unzip the downloaded files respectively.
   - For MIMIC-III: Place the raw data files in `mimic_datasets/mimic_iii/1.4/raw` directory, including `NOTEEVENTS.csv`, `ADMISSIONS.csv`, and `PATIENTS.csv`.
   - For MIMIC-IV EHR: Place `icu` and `hosp` folders of MIMIC-IV EHR data in `mimic_datasets/mimic_iv/3.1` directory, and unzip the `hosp/admissions.csv.gz`, `hosp/patients.csv.gz`, `icu/chartevents.csv.gz` and `icu/icustays.csv.gz` files.
   - For MIMIC-IV Notes: Place `note` folder of MIMIC-IV clinical notes data in `mimic_datasets/mimic_iv_note/2.2` directory, and unzip the `discharge.csv.gz` file.

3. Run Data Processing Scripts

   ```bash
   # Process MIMIC-III clinical notes data
   python mimic_iii_note.py

   # Process MIMIC-IV EHR data
   python mimic_iv_ehr.py

   # Process MIMIC-IV clinical notes data
   python mimic_iv_note.py
   ```

## Data Field Descriptions

### Patient Information

- PatientID: Unique patient identifier
- Age: Patient age
- Gender: Gender (0: Female, 1: Male)
- Race: Patient ethnicity/race

### Hospital Admission Information

- AdmissionID: Hospital admission record ID
- StayID: ICU stay record ID (MIMIC-IV only)
- AdmissionTime: Hospital admission time
- DischargeTime: Hospital discharge time
- DeathTime: Time of death (if applicable)
- ICUAdmissionTime: ICU admission time (MIMIC-IV only)
- ICUDischargeTime: ICU discharge time (MIMIC-IV only)
- LOS: Length of stay (hours)

### Clinical Events

- RecordTime: Record timestamp
- Variable: Clinical variable name
- Value: Clinical variable value

### Clinical Notes

- Text: Preprocessed clinical note text
- Category: Note type (Physician, Nursing, Nursing/other for MIMIC-III)
- RecordDate: Date of note creation
- RecordTime: Time of note creation

### Prediction Labels

- Outcome: In-hospital mortality
- InHospitalOutcome: Mortality during hospital stay
- InUnitOutcome: Mortality during ICU stay
- Readmission: 30-day readmission prediction

## Important Notes

1. Ensure you have obtained access to the MIMIC-III and MIMIC-IV databases before using this tool
2. Please comply with the MIMIC data usage agreements
3. It is recommended to have sufficient memory resources when processing large-scale data
4. MIMIC-III processing focuses on clinical notes data, while MIMIC-IV processing includes both EHR and clinical notes data
