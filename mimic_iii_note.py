import os
import re

import pandas as pd


data_dir = "mimic_datasets/mimic_iii/1.4/raw"
processed_dir = "mimic_datasets/mimic_iii/1.4/processed"
os.makedirs(processed_dir, exist_ok=True)

df = pd.read_csv(os.path.join(data_dir, "NOTEEVENTS.csv"), low_memory=False)
print(f"Total notes: {len(df)}")

# Remove duplicate notes
notes_df = df.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME"])
print(f"Notes after removing duplicate: {len(notes_df)}")

# Remove error notes
error_indice = notes_df[notes_df["ISERROR"].notna()]
notes_df = notes_df.drop(error_indice.index)
print(f"Notes after removing error: {len(notes_df)}")

# Keep only nursing, nursing/other, and physician notes
notes_df = notes_df[notes_df["CATEGORY"].isin(["Nursing", "Nursing/other", "Physician "])]
print(f"Notes after keeping only nursing, nursing/other, and physician notes: {len(notes_df)}")

# Remove notes with no admission ID
notes_df = notes_df[notes_df["HADM_ID"].notna()]
notes_df["HADM_ID"] = notes_df["HADM_ID"].astype(int)
print(f"Notes after removing notes with no admission ID: {len(notes_df)}")

# Rename columns and keep only the required columns
notes_df = notes_df.rename(columns={"SUBJECT_ID": "PatientID", "HADM_ID": "AdmissionID", "CHARTDATE": "RecordDate", "CHARTTIME": "RecordTime", "CATEGORY": "Category", "TEXT": "Text"})
notes_df = notes_df[["PatientID", "AdmissionID", "RecordDate", "RecordTime", "Category", "Text"]]

# Convert RecordDate and RecordTime to datetime and sort by PatientID, AdmissionID, RecordDate, RecordTime
notes_df["RecordDate"] = pd.to_datetime(notes_df["RecordDate"])
notes_df["RecordTime"] = pd.to_datetime(notes_df["RecordTime"])

# Define custom sorting order for Category
category_order = pd.CategoricalDtype(
    ['Physician ', 'Nursing', 'Nursing/other'],
    ordered=True
)
notes_df['Category'] = notes_df['Category'].astype(category_order)

# Sort by PatientID, AdmissionID, RecordDate, RecordTime
notes_df = notes_df.sort_values(by=["PatientID", "AdmissionID", "RecordDate", "Category", "RecordTime"]).reset_index(drop=True)

# Group by PatientID and AdmissionID and RecordDate and join the text
def create_formatted_text(group):
    all_category_blocks = []
    for category, notes_in_category in group.groupby("Category", observed=True):
        category_name = str(category).strip()
        header = f"Notes of type `{category_name}`:"
        content = "\n".join(notes_in_category["Text"])
        all_category_blocks.append(f"{header}\n{content}")
    return "\n".join(all_category_blocks)

notes_df_final = notes_df.groupby(
    ["PatientID", "AdmissionID", "RecordDate"]
)[["Category", "Text"]].apply(create_formatted_text).reset_index(name="Text")

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

# Group by PatientID and AdmissionID and take the first record
notes_df_final = notes_df_final.groupby(["PatientID", "AdmissionID"]).first().reset_index()
notes_df_final["Text"] = notes_df_final["Text"].apply(preprocess_text)

print(f"Number of patients in notes: {notes_df_final['PatientID'].nunique()}")
print(f"Number of admissions in notes: {notes_df_final['AdmissionID'].nunique()}")
print(f"Number of records in notes: {len(notes_df_final)}")

notes_df_final.to_parquet(os.path.join(processed_dir, "mimic_iii_note.parquet"), index=False)

admission_df = pd.read_csv(os.path.join(data_dir, "ADMISSIONS.csv"), low_memory=False)
admission_df = admission_df.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID"])
admission_df = admission_df.sort_values(by=["ROW_ID"]).reset_index(drop=True)[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "ETHNICITY"]]

patients_df = pd.read_csv(os.path.join(data_dir, "PATIENTS.csv"), low_memory=False)
patients_df = patients_df.drop_duplicates(subset=["ROW_ID", "SUBJECT_ID"])
patients_df = patients_df.sort_values(by=["ROW_ID"]).reset_index(drop=True)[["SUBJECT_ID", "GENDER", "DOB", "DOD"]]

merged_patients_df = pd.merge(admission_df, patients_df, on=["SUBJECT_ID"], how="inner")

merged_patients_df = merged_patients_df[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "DOD", "GENDER", "DOB", "ETHNICITY"]].rename(
    columns={
        "SUBJECT_ID": "PatientID",
        "HADM_ID": "AdmissionID",
        "ADMITTIME": "AdmissionTime",
        "DISCHTIME": "DischargeTime",
        "DEATHTIME": "DeathTime",
        "ETHNICITY": "Race",
        "GENDER": "Gender",
    }
)

def add_inhospital_mortality(df):
    mortality = df.DOD.notnull() & ((df.AdmissionTime <= df.DOD) & (df.DischargeTime >= df.DOD))
    mortality = mortality | (df.DeathTime.notnull() & ((df.AdmissionTime <= df.DeathTime) & (df.DischargeTime >= df.DeathTime)))
    df['InHospitalOutcome'] = mortality.astype(int)
    return df

def add_age(df):
    df.AdmissionTime = pd.to_datetime(df.AdmissionTime)
    df.DOB = pd.to_datetime(df.DOB)
    df['Age'] = df.AdmissionTime.dt.year - df.DOB.dt.year
    df['Age'] = df['Age'].apply(lambda x: 90 if x >= 90 or x <= 0 else int(x))
    return df

merged_patients_df = add_inhospital_mortality(merged_patients_df)
merged_patients_df = add_age(merged_patients_df)
merged_patients_df['Gender'] = merged_patients_df['Gender'].apply(lambda x: 1 if x == 'M' else 0)
merged_patients_df.to_parquet(os.path.join(processed_dir, "mimic_iii_patients.parquet"), index=False)

merged_patients_note_df = pd.merge(merged_patients_df, notes_df_final, on=["PatientID", "AdmissionID"], how="inner")
merged_patients_note_df.insert(0, "RecordID", merged_patients_note_df["PatientID"].astype(str) + "_" + merged_patients_note_df["AdmissionID"].astype(str))
merged_patients_note_df.to_parquet(os.path.join(processed_dir, "mimic_iii_note_label.parquet"), index=False)

print(f"Number of patients after merging with notes: {merged_patients_note_df['PatientID'].nunique()}")
print(f"Number of admissions after merging with notes: {merged_patients_note_df['AdmissionID'].nunique()}")
print(f"Number of records after merging with notes: {len(merged_patients_note_df)}")