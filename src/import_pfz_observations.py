from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from field_data_store import data_quality_issues, import_pfz_observations, normalize_pfz_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Import real PFZ observation rows into data/external/incois_pfz.csv")
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file to import.")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {args.csv_path}")

    result = import_pfz_observations(args.csv_path)
    imported_frame = normalize_pfz_dataset(pd.read_csv(args.csv_path))
    issues = data_quality_issues(imported_frame)

    print(f"Imported rows: {result['imported_rows']}")
    print(f"New rows added: {result['new_rows_added']}")
    print(f"Total rows now: {result['total_rows']}")
    if issues:
        print("Data quality notes:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("No data quality issues detected in the imported rows.")


if __name__ == "__main__":
    main()
