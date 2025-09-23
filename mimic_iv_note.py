import os

import tomli as tomllib
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # --- 1. Note Data Path ---
    note_data_dir = "mimic_datasets/mimic_iv_note/2.2"
    note_processed_dir = os.path.join(note_data_dir, "processed")
    os.makedirs(note_processed_dir, exist_ok=True)

    ehr_data_dir = "mimic_datasets/mimic_iv/3.1"

    # --- 2. Note Processing ---
    note_df = pd.read_csv(os.path.join(note_data_dir, "note", "discharge.csv"))
    note_saved_columns = ['subject_id', 'charttime', 'hadm_id', 'text']
    note_rename_map = {
        'subject_id': 'PatientID',
        'charttime': 'RecordTime',
        'hadm_id': 'AdmissionID',
        'text': 'Text'
    }
    note_df = note_df[note_saved_columns].rename(columns=note_rename_map)
    note_df = note_df.sort_values(by=['PatientID', 'RecordTime']).reset_index(drop=True)

    note_df["Text"] = (
        note_df["Text"]
        .str.replace(r'___+', '', regex=True)
        .str.replace(r'[^\w\s.,!?;]', '', regex=True)
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True).str.strip()
        .str.replace('name unit no admission date discharge date date of birth ', '', regex=False)
    )

    note_df.to_parquet(os.path.join(note_processed_dir,  "mimic4_discharge_note.parquet"), index=False)
    print(f"Discharge Note: {note_df['PatientID'].nunique()} patients with {note_df['AdmissionID'].nunique()} admissions and {note_df.shape[0]} notes.")

    # --- 3. Process EHR Data ---
    stays_path = os.path.join(ehr_data_dir, "processed", "mimic4_formatted_icustays.parquet")
    events_path = os.path.join(ehr_data_dir, "processed", "mimic4_formatted_events.parquet")
    config_file = os.path.join(ehr_data_dir, "config.toml")

    if not os.path.exists(events_path) or not os.path.exists(stays_path) or not os.path.exists(config_file):
        print(f"File events, stays or config_file does not exist. Exiting.")
        exit()

    stays_df = pd.read_parquet(stays_path)
    events_df = pd.read_parquet(events_path)
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    DAY_WINDOW = 7
    events_df['RecordTime'] = pd.to_datetime(events_df['RecordTime']).dt.date
    events_df = events_df.sort_values(by=['PatientID', 'AdmissionID', 'RecordTime'])

    # 1. Find unique dates for each admission
    unique_dates = events_df.drop_duplicates(subset=['PatientID', 'AdmissionID', 'RecordTime'])

    # 2. Use window function to get the (DAY_WINDOW + 1)th date from the last for each admission
    unique_dates['date_rank'] = unique_dates.groupby(['PatientID', 'AdmissionID'])['RecordTime'].rank(method='dense', ascending=False)
    split_dates = unique_dates[unique_dates['date_rank'] == DAY_WINDOW + 1][['PatientID', 'AdmissionID', 'RecordTime', 'date_rank']].rename(columns={'RecordTime': 'SplitTime'})

    # 3. Merge split points back to original events_df
    events_df = pd.merge(events_df, split_dates, on=['PatientID', 'AdmissionID'], how='left')

    # 4. Update RecordTime according to the condition
    # fillna is used to handle groups with less than (DAY_WINDOW + 1) days
    events_df['RecordTime'] = np.where(
        events_df['RecordTime'] >= events_df['SplitTime'].fillna(pd.Timestamp.max.date()),
        events_df['SplitTime'],
        events_df['RecordTime']
    )
    events_df = events_df.drop(columns=['SplitTime', 'date_rank'])

    # 5. Obtain the last record for each patient, admission, and record date
    events_df = events_df.groupby(['PatientID', 'AdmissionID', 'RecordTime']).last().reset_index()

    # --- 4. Merge Note with EHR Data ---
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
    df = df.rename(columns={"RecordTime_x": "NoteRecordTime", "RecordTime_y": "RecordTime"})
    df["RecordID"] = df["PatientID"].astype(str) + "_" + df["AdmissionID"].astype(str)

    # Rearrange the columns
    final_columns = ["RecordID", "PatientID", "RecordTime", "AdmissionID"] + config["label_features"] + config["demographic_features"] + config["labtest_features"] + ["NoteRecordTime", "Text"]
    df = df[final_columns]

    df = df.sort_values(by=['RecordID', 'RecordTime']).reset_index(drop=True)
    is_duplicate = df.duplicated(subset=['RecordID'], keep='first')
    df['Text'] = np.where(~is_duplicate, df['Text'], '')

    # --- 5. Save the final result ---
    df.to_parquet(os.path.join(note_processed_dir, "mimic4_discharge_note_ehr.parquet"), index=False)
    print(f"Discharge Note with EHR: {df['PatientID'].nunique()} patients with {df['AdmissionID'].nunique()} admissions and {df.shape[0]} records.")
    print("Data processing completed successfully.")