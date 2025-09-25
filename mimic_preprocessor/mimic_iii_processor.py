import os

import pandas as pd
from tqdm import tqdm

from mimic_preprocessor.utils import setup_logger, preprocess_text


class MIMICIIIProcessor:
    """
    A class to process the MIMIC-III dataset.
    """
    def __init__(self, data_dir: str, processed_dir: str, log_file: str = None):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.logger = setup_logger("MIMICIIIProcessor", log_file)
        tqdm.pandas(desc="Processing")

    def _process_notes(self) -> pd.DataFrame:
        """Processes the NOTEEVENTS.csv file."""
        self.logger.info("Starting processing of NOTEEVENTS.csv.")
        df = pd.read_csv(os.path.join(self.data_dir, "NOTEEVENTS.csv"), low_memory=False)
        self.logger.info(f"Total notes loaded: {len(df)}")

        df = df.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME"])
        self.logger.info(f"Notes after removing duplicates: {len(df)}")

        df = df.drop(df[df["ISERROR"].notna()].index)
        self.logger.info(f"Notes after removing error entries: {len(df)}")

        df = df[df["CATEGORY"].isin(["Nursing", "Nursing/other", "Physician "])]
        self.logger.info(f"Notes after filtering by category: {len(df)}")

        df = df[df["HADM_ID"].notna()]
        df["HADM_ID"] = df["HADM_ID"].astype(int)
        self.logger.info(f"Notes after removing entries with no admission ID: {len(df)}")

        df = df.rename(columns={"SUBJECT_ID": "PatientID", "HADM_ID": "AdmissionID", "CHARTDATE": "RecordDate", "CHARTTIME": "RecordTime", "CATEGORY": "Category", "TEXT": "Text"})
        df = df[["PatientID", "AdmissionID", "RecordDate", "RecordTime", "Category", "Text"]]

        df["RecordDate"] = pd.to_datetime(df["RecordDate"])
        df["RecordTime"] = pd.to_datetime(df["RecordTime"])

        category_order = pd.CategoricalDtype(['Physician ', 'Nursing', 'Nursing/other'], ordered=True)
        df['Category'] = df['Category'].astype(category_order)

        df = df.sort_values(by=["PatientID", "AdmissionID", "RecordDate", "Category", "RecordTime"]).reset_index(drop=True)

        def create_formatted_text(group):
            all_category_blocks = []
            for category, notes_in_category in group.groupby("Category", observed=True):
                header = f"Notes of type `{str(category).strip()}`:"
                content = "\n".join(notes_in_category["Text"])
                all_category_blocks.append(f"{header}\n{content}")
            return "\n".join(all_category_blocks)

        self.logger.info("Grouping notes by patient, admission, and date.")
        notes_df_grouped = df.groupby(["PatientID", "AdmissionID", "RecordDate"])[["Category", "Text"]]
        notes_df_final = notes_df_grouped.progress_apply(create_formatted_text).reset_index(name="Text")

        self.logger.info("Taking the first record per admission and preprocessing text.")
        notes_df_final = notes_df_final.groupby(["PatientID", "AdmissionID"]).first().reset_index()
        notes_df_final["Text"] = notes_df_final["Text"].progress_apply(preprocess_text)

        self.logger.info(f"Finished processing notes. Final counts: {len(notes_df_final)} records, {notes_df_final['AdmissionID'].nunique()} admissions, {notes_df_final['PatientID'].nunique()} patients.\n")
        notes_df_final.to_parquet(os.path.join(self.processed_dir, "mimic_iii_note.parquet"), index=False)
        return notes_df_final

    def _process_demographics(self) -> pd.DataFrame:
        """Processes the ADMISSIONS.csv and PATIENTS.csv files."""
        self.logger.info("Starting processing of ADMISSIONS.csv and PATIENTS.csv.")
        admission_df = pd.read_csv(os.path.join(self.data_dir, "ADMISSIONS.csv"), low_memory=False)
        admission_df = admission_df.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID"])
        admission_df = admission_df.sort_values(by=["ROW_ID"]).reset_index(drop=True)[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME"]]

        patients_df = pd.read_csv(os.path.join(self.data_dir, "PATIENTS.csv"), low_memory=False)
        patients_df = patients_df.drop_duplicates(subset=["ROW_ID", "SUBJECT_ID"])
        patients_df = patients_df.sort_values(by=["ROW_ID"]).reset_index(drop=True)[["SUBJECT_ID", "GENDER", "DOB", "DOD"]]

        merged_df = pd.merge(admission_df, patients_df, on=["SUBJECT_ID"], how="inner")
        merged_df = merged_df.rename(columns={
            "SUBJECT_ID": "PatientID", "HADM_ID": "AdmissionID", "ADMITTIME": "AdmissionTime",
            "DISCHTIME": "DischargeTime", "DEATHTIME": "DeathTime", "GENDER": "Gender"
        })
        merged_df = merged_df[["PatientID", "AdmissionID", "AdmissionTime", "DischargeTime", "DeathTime", "DOD", "Gender", "DOB"]]

        self.logger.info("Calculating in-hospital mortality and age.")
        merged_df.AdmissionTime = pd.to_datetime(merged_df.AdmissionTime, errors='coerce')
        merged_df.DischargeTime = pd.to_datetime(merged_df.DischargeTime, errors='coerce')
        merged_df.DeathTime = pd.to_datetime(merged_df.DeathTime, errors='coerce')
        merged_df.DOD = pd.to_datetime(merged_df.DOD, errors='coerce')

        mortality = merged_df.DOD.notnull() & (merged_df.AdmissionTime <= merged_df.DOD) & (merged_df.DischargeTime >= merged_df.DOD)
        mortality |= merged_df.DeathTime.notnull() & (merged_df.AdmissionTime <= merged_df.DeathTime) & (merged_df.DischargeTime >= merged_df.DeathTime)
        merged_df['InHospitalOutcome'] = mortality.astype(int)

        merged_df.DOB = pd.to_datetime(merged_df.DOB, errors='coerce')
        merged_df['Age'] = merged_df.AdmissionTime.dt.year - merged_df.DOB.dt.year
        merged_df['Age'] = merged_df['Age'].apply(lambda x: 90 if x >= 90 or x <= 0 else int(x))
        merged_df['Gender'] = merged_df['Gender'].apply(lambda x: 1 if x == 'M' else 0)

        self.logger.info("Finished processing demographics.")
        merged_df.to_parquet(os.path.join(self.processed_dir, "mimic_iii_patients.parquet"), index=False)
        return merged_df

    def process(self):
        """Executes the complete data processing pipeline."""
        self.logger.info("--- Starting MIMIC-III Data Processing ---")
        notes_df = self._process_notes()
        patients_df = self._process_demographics()

        self.logger.info("Merging notes and patient demographics data.")
        merged_final_df = pd.merge(patients_df, notes_df, on=["PatientID", "AdmissionID"], how="inner")
        merged_final_df.insert(0, "RecordID", merged_final_df["PatientID"].astype(str) + "_" + merged_final_df["AdmissionID"].astype(str))

        output_path = os.path.join(self.processed_dir, "mimic_iii_note_label.parquet")
        merged_final_df.to_parquet(output_path, index=False)

        self.logger.info(f"Final merged data saved to {output_path}")
        self.logger.info(f"Number of patients after merging: {merged_final_df['PatientID'].nunique()}")
        self.logger.info(f"Number of admissions after merging: {merged_final_df['AdmissionID'].nunique()}")
        self.logger.info(f"Total number of records: {len(merged_final_df)}")
        self.logger.info("--- MIMIC-III Data Processing Finished ---")