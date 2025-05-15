# MIMIC Preprocessor

This is an open-source tool for processing the MIMIC-IV (Medical Information Mart for Intensive Care IV) dataset. The tool is designed to process and transform MIMIC-IV's Electronic Health Records (EHR) and clinical notes data, making it more suitable for machine learning research.

## Dataset Source

- Structured EHR data: MIMIC-IV v3.1 (<https://physionet.org/content/mimiciv/3.1/>)
- Clinical notes data: MIMIC-IV v2.2 (<https://physionet.org/content/mimic-iv-note/2.2/>)

## Features

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
   - Sign up for MIMIC-IV access, download the data and unzip the downloaded files respectively.
   - Place `icu` and `hosp` folders of MIMIC-IV EHR data in `mimic_datasets/mimic_iv/3.1` directory, and unzip the `hosp/admissions.csv.gz`, `hosp/patients.csv.gz`, `icu/chartevents.csv.gz` and `icu/icustays.csv.gz` files.
   - Place `note` folder of MIMIC-IV clinical notes data in `mimic_datasets/mimic_iv_note/2.2` directory, and unzip the `discharge.csv.gz` file.

3. Run Data Processing Scripts

   ```bash
   # Process EHR data
   python mimic_iv_ehr.py

   # Process clinical notes data
   python mimic_iv_note.py
   ```

## Data Field Descriptions

### Patient Information

- PatientID: Unique patient identifier
- Age: Patient age
- Sex: Gender (0: Female, 1: Male)

### Hospital Admission Information

- AdmissionID: Hospital admission record ID
- StayID: ICU stay record ID
- AdmissionTime: Hospital admission time
- DischargeTime: Hospital discharge time
- ICUAdmissionTime: ICU admission time
- ICUDischargeTime: ICU discharge time
- LOS: Length of stay (hours)

### Clinical Events

- RecordTime: Record timestamp
- Variable: Clinical variable name
- Value: Clinical variable value

### Clinical Notes

- Text: Preprocessed clinical note text

### Prediction Labels

- Outcome: In-hospital mortality
- InHospitalOutcome: Mortality during hospital stay
- InUnitOutcome: Mortality during ICU stay
- Readmission: 30-day readmission prediction

## Important Notes

1. Ensure you have obtained access to the MIMIC-IV database before using this tool
2. Please comply with the MIMIC-IV data usage agreement
3. It is recommended to have sufficient memory resources when processing large-scale data
