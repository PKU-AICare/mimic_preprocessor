import os
import re

import tomli as tomllib
import pandas as pd


# Processing text function
def preprocess_text(text):
    # Replace sequences of underscores
    text = re.sub(r'___+', '', text)
    # Remove non-alphanumeric characters except punctuation (keep .,!?;)
    text = re.sub(r'[^\w\s.,!?;]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Strip extra white spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove de-identification placeholders
    text = text.replace('name unit no admission date discharge date date of birth ', '')
    return text


if __name__ == "__main__":
    note_data_dir = "mimic_datasets/mimic_iv_note/2.2"
    note_df = pd.read_csv(os.path.join(note_data_dir, "note", "discharge.csv"))
    note_processed_dir = os.path.join(note_data_dir, "processed")
    os.makedirs(note_processed_dir, exist_ok=True)

    note_saved_columns = ['subject_id', 'charttime', 'hadm_id', 'text']
    note_rename_map = {
        'subject_id': 'PatientID',
        'charttime': 'RecordTime',
        'hadm_id': 'AdmissionID',
        'text': 'Text'
    }
    note_df = note_df[note_saved_columns].rename(columns=note_rename_map).sort_values(by=['PatientID', 'RecordTime']).reset_index(drop=True)
    note_df["Text"] = note_df["Text"].apply(preprocess_text)
    note_df = note_df.groupby("PatientID").tail(5).reset_index(drop=True)

    note_df.to_parquet(os.path.join(note_processed_dir,  "mimic4_discharge_note.parquet"), index=False)

    # Merge with labels and events
    ehr_data_dir = "mimic_datasets/mimic_iv/3.1"
    stays = os.path.join(ehr_data_dir, "processed", "mimic4_formatted_stays.csv")
    events = os.path.join(ehr_data_dir, "processed", "mimic4_formatted_events.csv")
    config_file = os.path.join(ehr_data_dir, "config.toml")
    if not os.path.exists(config_file):
        print(f"File {config_file} does not exist. Exiting.")
        exit()
    config = tomllib.load(config_file)
    
    if not os.path.exists(stays):
        print(f"File {stays} does not exist. Exiting.")
        exit()
    stays_df = pd.read_csv(stays)
    note_df = note_df.merge(
        stays_df,
        how="inner",
        on=["PatientID", "AdmissionID"]
    )
    note_df[config["basic_features"] + ["Text"] + config["label_features"] + config["demographics_features"]].to_parquet(os.path.join(note_processed_dir, "mimic4_discharge_note_with_labels.parquet"), index=False)

    if not os.path.exists(events):
        print(f"File {events} does not exist. Exiting.")
        exit()
    events_df = pd.read_csv(events)
    note_df = note_df.merge(
        events_df,
        how="inner",
        on=["PatientID", "AdmissionID"]
    )
    note_df[config["basic_features"] + ["Text"] + config["label_features"] + config["demographics_features"] + config["labtest_features"]].to_parquet(os.path.join(note_processed_dir, "mimic4_discharge_note_with_events.parquet"), index=False)
    print("Data processing completed successfully.")