import os
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tomli as tomllib
from tqdm import tqdm

from mimic_preprocessor.utils import setup_logger, preprocess_text


class MIMICIVProcessor:
    """A class to process the MIMIC-IV dataset."""
    def __init__(self, data_dir: str, note_dir: str, processed_dir: str, log_file: Optional[str] = None, one_hot_encode_categorical: bool = False):
        self.data_dir = data_dir
        self.note_dir = note_dir
        self.processed_dir = processed_dir
        self.one_hot_encode_categorical = one_hot_encode_categorical
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(os.path.join(self.note_dir, "processed"), exist_ok=True)
        self.logger = setup_logger("MIMICIVProcessor", log_file)
        tqdm.pandas(desc="Processing")

    # --- Formatting Functions for Events EHR Data ---
    def _format_gcs(self, df): return df["valuenum"].astype(float)
    def _format_crr(self, df):
        v = pd.Series(np.nan, index=df.index)
        str_mask = df["value"].apply(isinstance, args=(str,))
        v.loc[str_mask & df["value"].str.contains('normal', case=False)] = 0.0
        v.loc[str_mask & df["value"].str.contains('abnormal', case=False)] = 1.0
        return v
    def _format_temperature(self, df):
        v = pd.to_numeric(df["value"], errors='coerce').astype(float)
        idx = df["MimicLabel"].str.contains('f', case=False)
        v.loc[idx] = (v[idx] - 32) * 5. / 9
        return v.round(2)
    def _format_weight(self, df):
        v = pd.to_numeric(df["value"], errors='coerce').astype(float)
        idx = df["MimicLabel"].str.contains('lbs', case=False)
        v.loc[idx] = v[idx] * 0.453592
        return v.round(2)
    def _format_height(self, df):
        v = pd.to_numeric(df["value"], errors='coerce').astype(float)
        idx = ~df["MimicLabel"].str.contains('cm', case=False)
        v.loc[idx] = v[idx] * 2.54
        return v.round()

    # --- One-Hot Encoding Method ---
    def _one_hot_encode_categorical_features(self, df, config):
        """Apply one-hot encoding to categorical labtest features.

        Args:
            df: DataFrame containing labtest features
            config: Configuration dictionary containing feature information

        Returns:
            DataFrame with one-hot encoded categorical features
        """
        if not self.one_hot_encode_categorical:
            return df

        self.logger.info("Applying one-hot encoding to categorical labtest features.")

        # Get categorical features from config
        categorical_features = [feature for feature, is_cat in config["is_categorical_channel"].items() if is_cat]
        possible_values = config["possible_values"]

        for feature in categorical_features:
            if feature in df.columns and possible_values.get(feature):
                # Create one-hot encoded columns
                for value_info in possible_values[feature]:
                    if isinstance(value_info, list):
                        # Handle tuple format: (int_value, description)
                        int_value, description = value_info
                        column_name = f"{feature}->{int_value} {description}"
                        # Handle both float and int values, including NaN
                        df[column_name] = (df[feature] == int_value).astype(int)
                    else:
                        # Handle string format (backward compatibility)
                        column_name = f"{feature}->{value_info}"
                        df[column_name] = (df[feature] == value_info).astype(int)

                # Remove original categorical column
                df = df.drop(columns=[feature])
                self.logger.info(f"One-hot encoded {feature} into {len(possible_values[feature])} columns")

        return df

    # --- Processing EHR Data ---
    def _process_icu_patients(self):
        """Processes patient, admission, and icustays tables to get ICU stays information.
            1. Merge patient, admission, and icustays tables.
            2. Calculate 30-day readmission, mortality in hospital and unit as label features.
            3. Format the data, including age, gender, and los, and rename the columns.

        Output:
            pd.DataFrame: ICU patients information with 30-day readmission, mortality in hospital and unit as label features.
        """

        if os.path.exists(os.path.join(self.processed_dir, "mimic4_formatted_patients.parquet")):
            self.logger.info("ICU patients data already processed. Loading from file.\n")
            return pd.read_parquet(os.path.join(self.processed_dir, "mimic4_formatted_patients.parquet"))

        # 1. Merge patient, admission, and icustays tables.
        self.logger.info("Processing patient, admission, and ICU stay information.")
        patients = pd.read_csv(os.path.join(self.data_dir, 'hosp', 'patients.csv'))[['subject_id', 'gender', 'anchor_age', 'dod']]
        admissions = pd.read_csv(os.path.join(self.data_dir, 'hosp', 'admissions.csv'))[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']]
        icustays = pd.read_csv(os.path.join(self.data_dir, 'icu', 'icustays.csv'))

        for df, cols in [(patients, ['dod']), (admissions, ['admittime', 'dischtime', 'deathtime']), (icustays, ['intime', 'outtime'])]:
            for col in cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        stays = icustays.merge(admissions, how="inner", on=["subject_id", "hadm_id"])
        stays = stays.merge(patients, how="inner", on=["subject_id"])

        # 2. Calculate 30-day readmission, mortality in hospital and unit as label features.
        self.logger.info("Calculating 30-day readmission.")
        stays = stays.sort_values(by=['subject_id', 'intime']).reset_index(drop=True)
        stays['next_intime'] = stays.groupby('subject_id')['intime'].shift(-1)
        days_to_next = (pd.to_datetime(stays['next_intime']) - pd.to_datetime(stays['outtime'])).dt.total_seconds() / (24 * 3600)
        days_to_death = (pd.to_datetime(stays['dod']) - pd.to_datetime(stays['outtime'])).dt.total_seconds() / (24 * 3600)
        readmit_cond = (days_to_next > 0) & (days_to_next <= 30) # Readmission within 30 days after discharge
        death_cond = stays['next_intime'].isna() & (days_to_death > 0) & (days_to_death <= 30) # Death within 30 days after discharge
        stays['readmission'] = (readmit_cond | death_cond).astype(int)
        stays = stays.drop(columns=['next_intime'])

        self.logger.info("Calculating in-hospital mortality.")
        mortality_inhospital = stays.dod.notnull() & (stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod)
        mortality_inhospital |= stays.deathtime.notnull() & (stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime)
        stays['mortality_inhospital'] = mortality_inhospital.astype(int)

        self.logger.info("Calculating in-unit mortality.")
        mortality_inunit = stays.dod.notnull() & (stays.intime <= stays.dod) & (stays.outtime >= stays.dod)
        mortality_inunit |= stays.deathtime.notnull() & (stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime)
        stays['mortality_inunit'] = mortality_inunit.astype(int)

        self.logger.info("Calculating mortality.")
        stays['mortality'] = (mortality_inunit | mortality_inhospital).astype(int)

        # 3. Format the data.
        stays['age'] = stays.anchor_age
        stays['gender'] = stays['gender'].apply(lambda s: 1 if s == 'M' else 0)
        stays['los'] = 24 * stays['los'].astype(float)

        rename_map = {
            'subject_id': 'PatientID', 'hadm_id': 'AdmissionID', 'stay_id': 'StayID', 'admittime': 'AdmissionTime',
            'dischtime': 'DischargeTime', 'deathtime': 'DeathTime', 'intime': 'ICUAdmissionTime', 'outtime': 'ICUDischargeTime',
            'dod': 'DeathDate', 'los': 'LOS', 'mortality_inunit': 'InUnitOutcome', 'mortality_inhospital': 'InHospitalOutcome',
            'age': 'Age', 'gender': 'Sex', 'readmission': 'Readmission', 'mortality': 'Outcome'
        }
        stays_processed = stays.rename(columns=rename_map)

        output_path = os.path.join(self.processed_dir, "mimic4_formatted_patients.parquet")
        stays_processed[list(rename_map.values())].to_parquet(output_path, index=False)
        self.logger.info(f"ICU patients data processed and saved to {output_path}.")

        self.logger.info("--- In ICU Patients Data ---")
        self.logger.info(f"Number of patients: {stays_processed['PatientID'].nunique()}")
        self.logger.info(f"Number of admissions: {stays_processed['AdmissionID'].nunique()}")
        self.logger.info(f"Number of records: {len(stays_processed)}")
        self.logger.info("--- In ICU Patients Data ---\n")

        return stays_processed

    def _process_events(self):
        """Processes the chartevents.csv file in chunks.
            1. Merge chartevents with item2var.csv to get the variable name.
            2. Format the features selected using the format functions.
            3. Pivot the data to wide format.

        Output:
            pd.DataFrame: Events data in wide format.
        """

        if os.path.exists(os.path.join(self.processed_dir, "mimic4_formatted_events.parquet")):
            self.logger.info("Events data already processed. Loading from file.\n")
            return pd.read_parquet(os.path.join(self.processed_dir, "mimic4_formatted_events.parquet"))

        self.logger.info("Processing events from chartevents.csv in chunks.")
        item2var_df = pd.read_csv(os.path.join(self.data_dir, "mimic4_item2var.csv"))
        with open(os.path.join(self.data_dir, "config.toml"), "rb") as f:
            config = tomllib.load(f)

        format_fns = {
            'Capillary refill rate': self._format_crr, 'Glascow coma scale motor response': self._format_gcs,
            'Glascow coma scale eye opening': self._format_gcs, 'Glascow coma scale verbal response': self._format_gcs,
            'Temperature': self._format_temperature, 'Weight': self._format_weight, 'Height': self._format_height
        }

        # Read chartevents in chunks and process them, and save to a temporary file.
        temp_path = os.path.join(self.processed_dir, "temp_events.parquet")
        if os.path.exists(temp_path): os.remove(temp_path)

        chartevents_iter = pd.read_csv(os.path.join(self.data_dir, "icu", "chartevents.csv"), chunksize=10_000_000, low_memory=False)
        writer = None

        for chunk in tqdm(chartevents_iter, total=44, desc="Processing event chunks"):
            df_chunk = chunk.merge(item2var_df, left_on='itemid', right_on='ItemID')
            df_chunk = df_chunk.drop_duplicates(subset=['subject_id', 'charttime', 'hadm_id', 'stay_id', 'Variable'], keep='last')

            for var_name, func in format_fns.items():
                idx = (df_chunk["Variable"] == var_name)
                if idx.any():
                    df_chunk.loc[idx, 'value'] = func(df_chunk.loc[idx])
            df_chunk['value'] = pd.to_numeric(df_chunk['value'], errors='coerce')
            df_chunk = df_chunk.dropna(subset=['value'])

            final_chunk = df_chunk.rename(columns={'subject_id': 'PatientID', 'charttime': 'RecordTime', 'hadm_id': 'AdmissionID', 'stay_id': 'StayID', 'value': 'Value'})
            table = pa.Table.from_pandas(final_chunk[['PatientID', 'RecordTime', 'AdmissionID', 'StayID', 'Variable', 'Value']])

            if writer is None: writer = pq.ParquetWriter(temp_path, table.schema)
            writer.write_table(table)

        if writer: writer.close()

        self.logger.info("Pivoting event data to wide format.")
        format_df = pd.read_parquet(temp_path)
        os.remove(temp_path)
        value_events = format_df.pivot_table(index=['PatientID', 'AdmissionID', 'StayID', 'RecordTime'], columns='Variable', values='Value', aggfunc='last').reset_index()

        gcs_cols = ["Glascow coma scale eye opening", "Glascow coma scale motor response", "Glascow coma scale verbal response"]
        for col in gcs_cols:
            if col not in value_events.columns: value_events[col] = np.nan
        value_events["Glascow coma scale total"] = value_events[gcs_cols].sum(axis=1)

        final_events = value_events[['PatientID', 'RecordTime', 'AdmissionID', 'StayID'] + config["labtest_features"]]
        output_path = os.path.join(self.processed_dir, "mimic4_formatted_events.parquet")
        final_events.to_parquet(output_path, index=False)
        self.logger.info(f"Events data processed and saved to {output_path}.")

        self.logger.info("--- In Events Data ---")
        self.logger.info(f"Number of patients: {final_events['PatientID'].nunique()}")
        self.logger.info(f"Number of admissions: {final_events['AdmissionID'].nunique()}")
        self.logger.info(f"Number of records: {len(final_events)}")
        self.logger.info("--- In Events Data ---\n")

        return final_events

    def _process_ehr_data(self, max_window: int = 7):
        """Integrates the EHR data processing pipeline.

        Args:
            aggregate_unit (Literal["day", "hour"]): The unit to aggregate the events data.
            max_window (int): The maximum window to aggregate the events data.
        """

        if os.path.exists(os.path.join(self.processed_dir, "mimic4_formatted_ehr.parquet")):
            self.logger.info("EHR data already processed. Loading from file.\n")
            return pd.read_parquet(os.path.join(self.processed_dir, "mimic4_formatted_ehr.parquet"))

        processed_patients = self._process_icu_patients()
        processed_events = self._process_events()

        with open(os.path.join(self.data_dir, "config.toml"), "rb") as f:
            config = tomllib.load(f)

        self.logger.info("Merging patients and events data.")
        ehr_df = processed_patients.merge(processed_events, on=['PatientID', 'AdmissionID', 'StayID'], how='inner')
        ehr_df = ehr_df[["PatientID", "RecordTime", "AdmissionID"] + config["label_features"] + config["demographic_features"] + config["labtest_features"]]

        self.logger.info(f"Aggregating events data by {max_window} days to events data.")
        ehr_df['RecordTime'] = pd.to_datetime(ehr_df['RecordTime']).dt.date
        ehr_df = ehr_df.sort_values(by=['PatientID', 'AdmissionID', 'RecordTime'])

        # Get the (max_window + 1)th date from the last for each admission
        unique_dates = ehr_df.drop_duplicates(subset=['PatientID', 'AdmissionID', 'RecordTime'])
        unique_dates.loc[:, 'date_rank'] = unique_dates.groupby(['PatientID', 'AdmissionID'])['RecordTime'].rank(method='dense', ascending=False).values
        split_dates = unique_dates[unique_dates['date_rank'] == max_window + 1][['PatientID', 'AdmissionID', 'RecordTime']].rename(columns={'RecordTime': 'SplitTime'})

        # Set all records before the (max_window + 1)th date to the (max_window + 1)th date, and get the last record by days for each patient, admission, and record time
        ehr_df = pd.merge(ehr_df, split_dates, on=['PatientID', 'AdmissionID'], how='left')
        ehr_df['RecordTime'] = np.where(ehr_df['RecordTime'] <= ehr_df['SplitTime'].fillna(pd.Timestamp.min.date()), ehr_df['SplitTime'], ehr_df['RecordTime'])
        ehr_df = ehr_df.drop(columns=['SplitTime']).groupby(['PatientID', 'AdmissionID', 'RecordTime']).last().reset_index()
        ehr_df["Glascow coma scale total"] = ehr_df[config["gcs_features"]].sum(axis=1)

        # Apply one-hot encoding to categorical features if enabled
        ehr_df = self._one_hot_encode_categorical_features(ehr_df, config)

        output_path = os.path.join(self.processed_dir, "mimic4_formatted_ehr.parquet")
        ehr_df.to_parquet(output_path, index=False)
        self.logger.info(f"Final EHR data saved to {output_path}.")

        self.logger.info("--- In EHR Data ---")
        self.logger.info(f"Number of patients: {ehr_df['PatientID'].nunique()}")
        self.logger.info(f"Number of admissions: {ehr_df['AdmissionID'].nunique()}")
        self.logger.info(f"Number of records: {len(ehr_df)}")
        self.logger.info("--- In EHR Data ---\n")

        return ehr_df

    def _process_note_data(self):
        """Processes discharge summary notes, following the Clinical-Longformer implementation."""
        if os.path.exists(os.path.join(self.note_dir, "processed", "mimic4_discharge_note.parquet")):
            self.logger.info("Discharge notes already processed. Loading from file.\n")
            return pd.read_parquet(os.path.join(self.note_dir, "processed", "mimic4_discharge_note.parquet"))

        self.logger.info("Processing discharge notes.")
        note_df = pd.read_csv(os.path.join(self.note_dir, "note", "discharge.csv"), low_memory=False)
        note_df = note_df.rename(columns={'subject_id': 'PatientID', 'charttime': 'RecordTime', 'hadm_id': 'AdmissionID', 'text': 'Text'})
        note_df = note_df[['PatientID', 'RecordTime', 'AdmissionID', 'Text']].sort_values(by=['PatientID', 'RecordTime']).reset_index(drop=True)
        note_df['Text'] = note_df['Text'].progress_apply(preprocess_text)

        output_path = os.path.join(self.note_dir, "processed", "mimic4_discharge_note.parquet")
        note_df.to_parquet(output_path, index=False)
        self.logger.info(f"Discharge notes processed and saved to {output_path}.")

        self.logger.info("--- In Note Data ---")
        self.logger.info(f"Number of patients: {note_df['PatientID'].nunique()}")
        self.logger.info(f"Number of admissions: {note_df['AdmissionID'].nunique()}")
        self.logger.info(f"Number of records: {len(note_df)}")
        self.logger.info("--- In Note Data ---\n")

        return note_df

    def _merge_ehr_and_note(self):
        """Merges the processed EHR and NOTE data."""
        if os.path.exists(os.path.join(self.processed_dir, "mimic4_discharge_note_ehr.parquet")):
            self.logger.info("Merged EHR and Note data already processed. Loading from file.\n")
            return pd.read_parquet(os.path.join(self.processed_dir, "mimic4_discharge_note_ehr.parquet"))

        self.logger.info("Starting merge of EHR and Note data.")
        note_df = pd.read_parquet(os.path.join(self.note_dir, "processed", "mimic4_discharge_note.parquet"))
        ehr_df = pd.read_parquet(os.path.join(self.processed_dir, "mimic4_formatted_ehr.parquet"))
        with open(os.path.join(self.data_dir, "config.toml"), "rb") as f:
            config = tomllib.load(f)

        # Merge EHR and Note data
        df = note_df.merge(ehr_df, how="inner", on=["PatientID", "AdmissionID"])
        df = df.rename(columns={"RecordTime_x": "NoteRecordTime", "RecordTime_y": "RecordTime"})

        # Add the record ID as the unique identifier
        df["RecordID"] = df["PatientID"].astype(str) + "_" + df["AdmissionID"].astype(str)

        # Determine labtest features based on whether one-hot encoding was applied
        if self.one_hot_encode_categorical:
            # Use one-hot labtest features from config if available
            if "one_hot_labtest_features" in config:
                labtest_feature_columns = [feature for feature, enabled in config["one_hot_labtest_features"].items() if enabled]
            else:
                # Fallback: dynamically determine one-hot features from the merged dataframe
                base_columns = ["RecordID", "PatientID", "RecordTime", "AdmissionID", "NoteRecordTime", "Text"] + config["label_features"] + config["demographic_features"]
                labtest_feature_columns = [col for col in df.columns if col not in base_columns]
        else:
            labtest_feature_columns = config["labtest_features"]

        final_columns = ["RecordID", "PatientID", "RecordTime", "AdmissionID"] + config["label_features"] + config["demographic_features"] + labtest_feature_columns + ["NoteRecordTime", "Text"]
        df = df[final_columns].sort_values(by=['RecordID', 'RecordTime']).reset_index(drop=True)
        df['Text'] = np.where(~df.duplicated(subset=['RecordID'], keep='first'), df['Text'], '')

        output_path = os.path.join(self.processed_dir, "mimic4_discharge_note_ehr.parquet")
        df.to_parquet(output_path, index=False)
        self.logger.info(f"Merged EHR and Note data saved to {output_path}.")

        self.logger.info("--- In Merged EHR and Note Data ---")
        self.logger.info(f"Number of patients: {df['PatientID'].nunique()}")
        self.logger.info(f"Number of admissions: {df['AdmissionID'].nunique()}")
        self.logger.info(f"Number of records: {len(df)}")
        self.logger.info("--- In Merged EHR and Note Data ---\n")

        return df

    def _process_icd_data(self):
        """Processes ICD codes and prescription data based on existing records."""
        self.logger.info("Processing ICD and prescription data.")
        merged_ehr_note_path = os.path.join(self.processed_dir, "mimic4_discharge_note_ehr.parquet")
        if not os.path.exists(merged_ehr_note_path):
            raise FileNotFoundError("Merged EHR/Note data not found. Please run the merge process first by using the --merge flag.\n")

        mimic4_ehr_note = pd.read_parquet(merged_ehr_note_path)
        record_ids = set(mimic4_ehr_note["RecordID"])

        for file in ["diagnoses_icd", "procedures_icd", "prescriptions"]:
            if os.path.exists(os.path.join(self.processed_dir, f"mimic4_{file}.parquet")):
                self.logger.info(f"{file} data already processed. Loading from file.\n")
                continue

            self.logger.info(f"Processing {file}.csv")
            df = pd.read_csv(os.path.join(self.data_dir, "hosp", f"{file}.csv"), low_memory=False)
            df = df.rename(columns={"subject_id": "PatientID", "hadm_id": "AdmissionID"})
            df["RecordID"] = df["PatientID"].astype(str) + "_" + df["AdmissionID"].astype(str)

            selected_df = df[df["RecordID"].isin(record_ids)].copy()
            if file == "prescriptions":
                selected_df.loc[:, 'gsn'] = selected_df['gsn'].astype(str)

            output_path = os.path.join(self.processed_dir, f"mimic4_{file}.parquet")
            selected_df.to_parquet(output_path, index=False)
            self.logger.info(f"Processed {file} data saved. Records found: {len(selected_df)}.")

    def process(self, parts: List[str], merge_ehr_note: bool = False):
        """
        Executes the main processing pipeline for MIMIC-IV.

        Args:
            parts (List[str]): The parts to process ('ehr', 'note', 'icd').
            merge_ehr_note (bool): Whether to merge the EHR and Note data.
        """
        self.logger.info(f"--- Starting MIMIC-IV Processing for parts: {parts} ---")
        if "ehr" in parts:
            self._process_ehr_data()

        if "note" in parts:
            self._process_note_data()

        if merge_ehr_note:
            if not all(os.path.exists(p) for p in [os.path.join(self.processed_dir, "mimic4_formatted_ehr.parquet"), os.path.join(self.note_dir, "processed", "mimic4_discharge_note.parquet")]):
                self.logger.warning("To merge, both processed EHR and Note files must exist. Please process them first. Skipping merge.")
            else:
                self._merge_ehr_and_note()

        if "icd" in parts:
            self._process_icd_data()

        self.logger.info("--- MIMIC-IV Data Processing Finished ---")