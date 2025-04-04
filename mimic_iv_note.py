import os
import re

import tomli as tomllib
import pandas as pd
import numpy as np


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


def last_not_null(series: pd.Series):
    non_null = series[series.notna()]
    return non_null.iloc[-1] if not non_null.empty else np.nan


def merge_by_days(df: pd.DataFrame, day: int=7):
    df = df.sort_values(by=['RecordTime']).reset_index(drop=True)
    days = df['RecordTime'].drop_duplicates().tolist()
    if len(days) > day:
        split_day = days[-day - 1]
        df.loc[df['RecordTime'] <= split_day, 'RecordTime'] = split_day
    df = df.groupby(['RecordTime']).agg(last_not_null).reset_index()
    return df


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
    print(f"Discharge Note: {note_df['PatientID'].nunique()} patients with {note_df.shape[0]} notes.")
    print("Note data processing completed successfully.")

    # Merge with labels and events
    ehr_data_dir = "mimic_datasets/mimic_iv/3.1"
    stays = os.path.join(ehr_data_dir, "processed", "mimic4_formatted_icustays.parquet")
    events = os.path.join(ehr_data_dir, "processed", "mimic4_formatted_events.parquet")
    config_file = os.path.join(ehr_data_dir, "config.toml")

    if not os.path.exists(events) or not os.path.exists(stays) or not os.path.exists(config_file):
        print(f"File events or stays or config_file does not exist. Exiting.")
        exit()
    
    stays_df = pd.read_parquet(stays)
    events_df = pd.read_parquet(events)
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    # Merge by AdmissionID
    # events_df = events_df.groupby(["AdmissionID"]).agg(last_not_null).reset_index()
    
    # Merge by days in one admission
    events['RecordTime'] = pd.to_datetime(events['RecordTime'])
    df = events.groupby(['PatientID', 'AdmissionID']).apply(merge_by_days, day=7).reset_index(drop=True)
    events_df = events_df.sort_values(by=["PatientID", "RecordTime"]).reset_index(drop=True)
    
    ehr_df = stays_df.merge(
        events_df,
        how="inner",
        on=["PatientID", "AdmissionID", "StayID"]
    )
    df = note_df.merge(
        ehr_df,
        how="inner",
        on=["PatientID", "AdmissionID"]
    )
    df = df.rename(columns={"RecordTime_x": "RecordTime",})
    df = df[["PatientID", "RecordTime", "AdmissionID"] + config["label_features"] + ["Text"] + config["demographic_features"] + config["labtest_features"]]
    
    df.to_parquet(os.path.join(note_processed_dir, "mimic4_discharge_note_ehr.parquet"), index=False)
    print(f"Discharge Note with EHR: {df['PatientID'].nunique()} patients with {df['AdmissionID'].nunique()} notes and {df.shape[0]} records.")
    print("Data processing completed successfully.")