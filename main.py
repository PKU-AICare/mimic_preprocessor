import argparse

from mimic_preprocessor.mimic_iii_processor import MIMICIIIProcessor
from mimic_preprocessor.mimic_iv_processor import MIMICIVProcessor


def main():
    parser = argparse.ArgumentParser(description="MIMIC Dataset Processor")
    parser.add_argument('dataset', choices=['mimic3', 'mimic4'], help="The dataset to process.")

    # MIMIC-III arguments
    parser.add_argument('--mimic3_data_dir', type=str, default="mimic_datasets/mimic_iii/1.4/raw", help="Path to raw MIMIC-III data.")
    parser.add_argument('--mimic3_processed_dir', type=str, default="mimic_datasets/mimic_iii/1.4/processed", help="Path to save processed MIMIC-III data.")

    # MIMIC-IV arguments
    parser.add_argument('--mimic4_ehr_dir', type=str, default="mimic_datasets/mimic_iv/3.1", help="Path to raw MIMIC-IV EHR data.")
    parser.add_argument('--mimic4_note_dir', type=str, default="mimic_datasets/mimic_iv_note/2.2", help="Path to raw MIMIC-IV Note data.")
    parser.add_argument('--mimic4_processed_dir', type=str, default="mimic_datasets/mimic_iv/3.1/processed", help="Path to save processed MIMIC-IV data.")
    parser.add_argument('--parts', nargs='+', choices=['ehr', 'note', 'icd'], default=['ehr', 'note', 'icd'], help="For MIMIC-IV, specify which parts to process.")
    parser.add_argument('--merge', action='store_true', help="For MIMIC-IV, merge EHR and Note data after processing.")

    # General arguments
    parser.add_argument('--log_file', type=str, help="Optional file to write logs to. Defaults to console output.")

    args = parser.parse_args()

    if args.dataset == 'mimic3':
        processor = MIMICIIIProcessor(
            data_dir=args.mimic3_data_dir,
            processed_dir=args.mimic3_processed_dir,
            log_file=args.log_file
        )
        processor.process()

    elif args.dataset == 'mimic4':
        processor = MIMICIVProcessor(
            data_dir=args.mimic4_ehr_dir,
            note_dir=args.mimic4_note_dir,
            processed_dir=args.mimic4_processed_dir,
            log_file=args.log_file
        )
        processor.process(parts=args.parts, merge_ehr_note=args.merge)


if __name__ == '__main__':
    main()