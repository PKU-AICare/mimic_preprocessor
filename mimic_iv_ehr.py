import os

import tomli as tomllib
import numpy as np
import pandas as pd
from pandas import Series


def read_patients_table(path):
    df = pd.read_csv(os.path.join(path, 'hosp', 'patients.csv'))
    df = df[['subject_id', 'gender', 'anchor_age', 'dod']]
    df.dod = pd.to_datetime(df.dod)
    return df


def read_admissions_table(path):
    df = pd.read_csv(os.path.join(path, 'hosp', 'admissions.csv'))
    df = df[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']]
    df.admittime = pd.to_datetime(df.admittime)
    df.dischtime = pd.to_datetime(df.dischtime)
    df.deathtime = pd.to_datetime(df.deathtime)
    return df


def read_icustays_table(path):
    df = pd.read_csv(os.path.join(path, 'icu', 'icustays.csv'))
    df.intime = pd.to_datetime(df.intime)
    df.outtime = pd.to_datetime(df.outtime)
    return df


def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.dod.notnull() & ((stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & ((stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime)))
    stays['mortality_inhospital'] = mortality.astype(int)
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.dod.notnull() & ((stays.intime <= stays.dod) & (stays.outtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & ((stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime)))
    stays['mortality'] = mortality.astype(int)
    stays['mortality_inunit'] = mortality.astype(int)
    return stays


def add_inunit_readmission_to_icustays(stays):

    def td_within_30days(datetime1, datetime2):
        datetime1 = pd.to_datetime(datetime1)
        datetime2 = pd.to_datetime(datetime2)
        return 1 if (datetime2 - datetime1).days <= 30 else 0

    patients = stays.groupby('subject_id')
    for _, group in patients:
        stays.loc[group.index, 'readmission'] = 0.0
        for i, group_row in enumerate(group.iterrows()):
            idx, row = group_row
            if i == len(group) - 1 and row.dod is not None:
                stays.loc[idx, 'readmission'] = td_within_30days(row.intime, row.dod)
            elif i < len(group) - 1:
                stays.loc[idx, 'readmission'] = td_within_30days(row.outtime, group.iloc[i + 1].intime)
    stays['readmission'] = stays['readmission'].astype(int)
    return stays


def format_gcs(df):
    return df["valuenum"].astype(float).copy()


def format_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan
    neg_idx = df["Value"].apply(lambda s: 'normal' in s.lower())
    pos_idx = df["Value"].apply(lambda s: 'abnormal' in s.lower())
    v.loc[neg_idx] = 0.0
    v.loc[pos_idx] = 1.0
    return v


def format_temperature(df):
    v = df["Value"].astype(float).copy()
    idx = df["MimicLabel"].apply(lambda s: 'F' in s.lower())
    v.loc[idx] = (v[idx] - 32) * 5. / 9
    return v


def format_weight(df):
    v = df["Value"].astype(float).copy()
    idx = df["MimicLabel"].apply(lambda s: 'lbs' in s.lower())
    v.loc[idx] = np.round(v[idx] * 0.453592, 1)
    return v


def format_height(df):
    v = df["Value"].astype(float).copy()
    idx = df["MimicLabel"].apply(lambda s: 'cm' not in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v


format_fns = {
    'Capillary refill rate': format_crr,
    "Glascow coma scale motor response": format_gcs,
    "Glascow coma scale eye opening": format_gcs,
    "Glascow coma scale verbal response": format_gcs,
    'Temperature': format_temperature,
    'Weight': format_weight,
    'Height': format_height
}


def format_events(events):
    global format_fns
    for var_name, format_fn in format_fns.items():
        idx = (events["Variable"] == var_name)
        events.loc[idx, 'Value'] = format_fn(events[idx])
    return events.loc[events["Value"].notnull()]


if __name__ == '__main__':
    data_dir = "mimic_datasets/mimic_iv/3.1"
    patients = os.path.join(data_dir, "hosp", "patients.csv")
    admissions = os.path.join(data_dir, "hosp", "admissions.csv")
    icustays = os.path.join(data_dir, "icu", "icustays.csv")
    chartevents = os.path.join(data_dir, "icu", "chartevents.csv")
    item2var = os.path.join(data_dir, "mimic4_item2var.csv")
    config_file = os.path.join(data_dir, "config.toml")
    
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    ## Processing patients information
    patients = read_patients_table(data_dir)
    admissions = read_admissions_table(data_dir)
    icustays = read_icustays_table(data_dir)

    print("Patients: ", len(patients))
    print(f"Admissions: {patients.subject_id.nunique()} patients, {len(admissions)} records")
    print(f"ICU stays: {icustays.subject_id.nunique()} patients, {len(icustays)} records")
    
    stays = icustays.merge(admissions, how="inner", on=["subject_id", "hadm_id"])
    stays = stays.merge(patients, how="inner", on=["subject_id"])
    print(f"ICU stays after merging with admissions and patients: {stays.subject_id.nunique()} patients, {len(stays)} records")
    
    stays = stays.sort_values(by=['subject_id', 'intime']).reset_index(drop=True)
    stays['age'] = stays.anchor_age
    stays.loc[:, 'gender'] = stays['gender'].apply(lambda s: 1 if s == 'M' else 0)
    stays = add_inhospital_mortality_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    stays = add_inunit_readmission_to_icustays(stays)
    stays['los'] = 24 * stays['los'].astype(float)
    stays_saved_columns = [
        'subject_id', 'hadm_id', 'stay_id', 
        'admittime', 'dischtime', 'deathtime', 'intime', 'outtime', 'dod',
        'mortality', 'los', 'readmission', 'mortality_inhospital', 'mortality_inunit', 
        'age', 'gender'
    ]
    stays_rename_map = {
        'subject_id': 'PatientID',
        'hadm_id': 'AdmissionID',
        'stay_id': 'StayID',
        'admittime': 'AdmissionTime',
        'dischtime': 'DischargeTime',
        'deathtime': 'DeathTime',
        'intime': 'ICUAdmissionTime',
        'outtime': 'ICUDischargeTime',
        'dod': 'DeathDate',
        'mortality': 'Outcome',
        'los': 'LOS',
        'readmission': 'Readmission',
        'mortality_inhospital': 'InHospitalOutcome',
        'mortality_inunit': 'InUnitOutcome',
        'age': 'Age',
        'gender': 'Sex'
    }
    stays = stays[stays_saved_columns].rename(columns=stays_rename_map)
    stays.to_csv(os.path.join(processed_dir, "mimic4_formatted_icustays.csv"), index=False)
    print(f"Done processing patients and admissions information. {stays.PatientID.nunique()} patients, {len(stays)} records")
    
    ## Processing events information
    print("Reading events tables...")
    chartevents_df = pd.read_csv(chartevents)
    print(f"Events: {chartevents_df.subject_id.nunique()} patients")
    item2var_df = pd.read_csv(item2var)
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    events_saved_columns = ['subject_id', 'charttime', 'hadm_id', 'stay_id', 'Variable', 'value']
    events_rename_map = {
        'subject_id': 'PatientID',
        'charttime': 'RecordTime',
        'hadm_id': 'AdmissionID',
        'stay_id': 'StayID',
        'value': 'Value',
    }
    df = chartevents_df.merge(item2var_df, left_on='itemid', right_on='ItemID')
    df = df[events_saved_columns + ['MimicLabel', 'valuenum']].rename(columns=events_rename_map)
    df = df.drop_duplicates(subset=['PatientID', 'RecordTime', 'AdmissionID', 'StayID', 'Variable'], keep='last')
    
    format_df = format_events(df)
    
    meta_events = format_df[['PatientID', 'RecordTime', 'AdmissionID', 'StayID']].sort_values(by=['PatientID', 'RecordTime', 'AdmissionID', 'StayID']).drop_duplicates(keep='first').set_index('RecordTime')
    value_events = format_df[['RecordTime', 'Variable', 'Value']].sort_values(by=['RecordTime', 'Variable', 'Value']).drop_duplicates(subset=['RecordTime', 'Variable'], keep='last')
    value_events = value_events.pnote_dfot(index='RecordTime', columns='Variable', values='Value')
    merged_events = meta_events.merge(value_events, left_index=True, right_index=True).sort_index(axis=0).sort_values(by=['PatientID', 'RecordTime', 'AdmissionID', 'StayID']).reset_index()
    merged_events = merged_events[['PatientID', 'RecordTime'] + list(merged_events.columns)[2:]]
    
    merged_events["Glascow coma scale total"] = merged_events["Glascow coma scale eye opening"].astype(float) + merged_events["Glascow coma scale motor response"].astype(float) + merged_events["Glascow coma scale verbal response"].astype(float)
    
    final_events = merged_events[['PatientID', 'RecordTime', 'AdmissionID', 'StayID'] + config["labtest_features"]]
    final_events.to_csv(os.path.join(processed_dir, "mimic4_formatted_events.csv"), index=False)
    print(f"Done processing events information. {final_events.PatientID.nunique()} patients, {len(final_events)} records")
    
    ## Mergeing events with stays
    print("Merging events with stays...")
    basic_features = ['PatientID', 'RecordTime']
    label_features = ['Outcome', 'LOS', 'Readmission']
    demo_features = ['Age', 'Sex']
    labevents_features = [
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale verbal response",
        "Glascow coma scale total",    
        "Capillary refill rate",
        "Diastolic blood pressure",
        "Fraction inspired oxygen",
        "Glucose",
        "Heart Rate",
        "Height",
        "Mean blood pressure",
        "Oxygen saturation",
        "Respiratory rate",
        "Systolic blood pressure",
        "Temperature",
        "Weight",
        "pH"
    ]
    merged_df = stays.merge(final_events, on=['PatientID', 'AdmissionID', 'StayID'], how='inner')
    saved_df = merged_df[config["basic_features"] + config["label_features"] + config["demographic_features"] + config["labtest_features"]]
    saved_df.to_csv(os.path.join(processed_dir, "mimic4_formatted_ehr.csv"), index=False)
    print(f"Done processing all information. {saved_df.PatientID.nunique()} patients, {len(saved_df)} records")