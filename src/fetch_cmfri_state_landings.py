from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "cmfri_state_landings.csv"
YEARS = range(2013, 2025)

STATE_NAME_MAP = {
    "Orissa": "Odisha",
    "Tamilnadu": "Tamil Nadu",
    "Pondicherry": "Puducherry",
}

VALID_STATES = {
    "West Bengal",
    "Odisha",
    "Andhra Pradesh",
    "Tamil Nadu",
    "Puducherry",
    "Kerala",
    "Karnataka",
    "Goa",
    "Maharashtra",
    "Gujarat",
}


def normalize_state(state_name: str) -> str:
    return STATE_NAME_MAP.get(state_name.strip(), state_name.strip())


def parse_year(year: int) -> list[dict[str, object]]:
    url = f"https://www.cmfri.org.in/state{year}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    pairs = re.findall(r"<td[^>]*>([^<]+)</td>\s*<td[^>]*>([0-9,]+)</td>", response.text, flags=re.IGNORECASE)
    rows: list[dict[str, object]] = []
    seen_states: set[str] = set()

    for raw_state, raw_value in pairs:
        state = normalize_state(raw_state)
        if state not in VALID_STATES or state in seen_states:
            continue

        rows.append(
            {
                "Year": year,
                "State": state,
                "Landings_Tonnes": int(raw_value.replace(",", "")),
                "Source_URL": url,
            }
        )
        seen_states.add(state)

    if len(rows) < 8:
        raise ValueError(f"Could not parse enough state rows from {url}")

    return rows


def main() -> None:
    all_rows: list[dict[str, object]] = []
    for year in YEARS:
        year_rows = parse_year(year)
        all_rows.extend(year_rows)
        print(f"Parsed {len(year_rows)} state rows for {year}")

    frame = pd.DataFrame(all_rows).sort_values(["State", "Year"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved CMFRI landings to {OUTPUT_PATH}")
    print(frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
