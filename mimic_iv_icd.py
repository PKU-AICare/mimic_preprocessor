import os
import pandas as pd

data_dir = "mimic_datasets/mimic_iv/3.1"
hosp_dir = os.path.join(data_dir, "hosp")
processed_dir = os.path.join(data_dir, "processed")

mimic4_ehr = pd.read_parquet(os.path.join(processed_dir, "mimic4_discharge_note_ehr.parquet"))

print("In MIMIC4 EHR and Note Dataset:")
print(f"Number of records: {len(mimic4_ehr)}")
print(f"Number of unique records: {mimic4_ehr['RecordID'].nunique()}")
print(f"Number of unique patients: {mimic4_ehr['PatientID'].nunique()}")
print("--------------------------------")

for file in ["diagnoses_icd", "procedures_icd", "prescriptions"]:
    df = pd.read_csv(os.path.join(hosp_dir, f"{file}.csv"), low_memory=False)
    df = df.rename(columns={"subject_id": "PatientID", "hadm_id": "AdmissionID"})
    df.insert(0, "RecordID", df["PatientID"].astype(str) + "_" + df["AdmissionID"].astype(str))

    selected_df = df[df["RecordID"].isin(mimic4_ehr["RecordID"])]
    print(f"Number of patients in selected {file}: {selected_df['PatientID'].nunique()}")
    print(f"Number of unique records in selected {file}: {selected_df['RecordID'].nunique()}")
    print(f"Number of all records in selected {file}: {len(selected_df)}")
    print(f"Average number of records per patient in selected {file}: {len(selected_df) / selected_df['RecordID'].nunique()}\n")

    selected_mimic4_ehr = mimic4_ehr[mimic4_ehr["RecordID"].isin(selected_df["RecordID"])]
    selected_mimic4_ehr = selected_mimic4_ehr[["RecordID", "Outcome", "LOS", "Readmission"]].groupby("RecordID").first().reset_index()
    merged_label_df = pd.merge(selected_df, selected_mimic4_ehr, on="RecordID", how="left")
    basic_columns = ["RecordID", "PatientID", "AdmissionID", "Outcome", "LOS", "Readmission"]
    all_columns = basic_columns + list(set(df.columns) - set(basic_columns))
    merged_label_df = merged_label_df[all_columns]

    print(f"Number of unique patients in merged {file}: {merged_label_df['PatientID'].nunique()}")
    print(f"Number of unique records in merged {file}: {merged_label_df['RecordID'].nunique()}")
    print(f"Number of all records in merged {file}: {len(merged_label_df)}")
    print(f"Average number of records per patient in merged {file}: {len(merged_label_df) / merged_label_df['RecordID'].nunique()}\n")

    if file == "prescriptions":
        merged_label_df['gsn'] = merged_label_df['gsn'].astype(str)
    merged_label_df.to_parquet(os.path.join(processed_dir, f"mimic4_{file}.parquet"))