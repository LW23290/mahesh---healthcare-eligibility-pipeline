#!/usr/bin/env python3
"""
Healthcare Eligibility Pipeline (Config-Driven)

What this script does:
1) Reads partner files using settings from a YAML config (delimiter + column mapping)
2) Maps partner columns -> standardized schema
3) Applies required transformations (name casing, dob format, email lower, phone formatting)
4) Writes ONE unified output file + an optional error report

Key idea:
Adding a new partner should only require updating configs/partners.yaml (no code changes).
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# -----------------------------
# Standard Schema (final output)
# -----------------------------
STANDARD_FIELDS = [
    "external_id",
    "first_name",
    "last_name",
    "dob",
    "email",
    "phone",
    "partner_code",
]


# -----------------------------
# Config models
# -----------------------------
@dataclass(frozen=True)
class PartnerConfig:
    partner_key: str
    partner_code: str
    delimiter: str
    header: bool
    column_mapping: Dict[str, str]  # partner_column -> standard_field


def load_configs(config_path: str) -> Dict[str, PartnerConfig]:
    """Loads partner configs from YAML and converts them into PartnerConfig objects."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(path.read_text())
    partners = raw.get("partners", {})

    configs: Dict[str, PartnerConfig] = {}
    for partner_key, p in partners.items():
        fmt = p.get("file_format", {})
        if fmt.get("type") != "delimited":
            raise ValueError(f"Partner '{partner_key}' has unsupported file_format.type: {fmt.get('type')}")

        configs[partner_key] = PartnerConfig(
            partner_key=partner_key,
            partner_code=str(p["partner_code"]),
            delimiter=str(fmt.get("delimiter", ",")),
            header=bool(fmt.get("header", True)),
            column_mapping=dict(p.get("column_mapping", {})),
        )

    return configs


# -----------------------------
# Transform helpers
# -----------------------------
def title_case(value: Optional[str]) -> Optional[str]:
    """Converts names to Title Case (safe for empty values)."""
    if not value:
        return None
    value = value.strip()
    return value.title() if value else None


def lower_case(value: Optional[str]) -> Optional[str]:
    """Lowercases emails safely."""
    if not value:
        return None
    value = value.strip()
    return value.lower() if value else None


def parse_dob(value: Optional[str]) -> Optional[str]:
    """
    Converts DOB into ISO format YYYY-MM-DD.
    Supports common partner formats:
    - MM/DD/YYYY
    - YYYY-MM-DD
    """
    if not value:
        return None

    value = value.strip()
    if not value:
        return None

    known_formats = ["%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%Y/%m/%d"]
    for fmt in known_formats:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue

    # If date is invalid, return None (we will log it as a data quality issue if needed)
    return None


def format_phone(value: Optional[str]) -> Optional[str]:
    """
    Normalizes phone to XXX-XXX-XXXX.
    Accepts:
      - "5551234567"
      - "555-222-3333"
      - "(555) 222 3333"
      - "1-555-222-3333" (uses last 10 digits)
    """
    if not value:
        return None

    digits = re.sub(r"\D+", "", value)
    if len(digits) < 10:
        return None

    digits = digits[-10:]  # last 10 digits is typical for US phone
    return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"


# -----------------------------
# Core pipeline functions
# -----------------------------
def read_rows(file_path: str, delimiter: str, has_header: bool) -> Tuple[List[str], List[List[str]]]:
    """
    Reads a delimited file and returns (header_columns, rows).
    We keep this very simple and robust for the assessment.
    """
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        all_rows = [row for row in reader if row and any(cell.strip() for cell in row)]
        if not all_rows:
            return [], []

        if has_header:
            header = [h.strip() for h in all_rows[0]]
            rows = all_rows[1:]
            return header, rows

        # no header scenario (not needed for this assessment, but included)
        header = [f"col_{i}" for i in range(len(all_rows[0]))]
        return header, all_rows


def map_to_standard(header: List[str], row: List[str], cfg: PartnerConfig) -> Dict[str, Optional[str]]:
    """
    Converts partner row into standardized dict based on config mapping.
    """
    partner_dict = {header[i]: (row[i].strip() if i < len(row) else "") for i in range(len(header))}

    standard = {k: None for k in STANDARD_FIELDS}

    for partner_col, standard_field in cfg.column_mapping.items():
        if standard_field in standard:
            standard[standard_field] = partner_dict.get(partner_col)

    standard["partner_code"] = cfg.partner_code
    return standard


def apply_transformations(record: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Applies all required transformations to the standardized record."""
    record["external_id"] = (record.get("external_id") or "").strip() or None
    record["first_name"] = title_case(record.get("first_name"))
    record["last_name"] = title_case(record.get("last_name"))
    record["dob"] = parse_dob(record.get("dob"))
    record["email"] = lower_case(record.get("email"))
    record["phone"] = format_phone(record.get("phone"))
    return record


def validate(record: Dict[str, Optional[str]]) -> List[str]:
    """
    Returns a list of validation errors. Empty list means valid.
    Bonus:
      - external_id must exist
    """
    errors: List[str] = []

    if not record.get("external_id"):
        errors.append("missing_external_id")

    return errors


def process_partner(partner_key: str, input_file: str, cfg: PartnerConfig, drop_invalid: bool) -> Tuple[List[Dict], List[Dict]]:
    """
    Processes one partner file into standardized records + error logs.
    """
    header, rows = read_rows(input_file, cfg.delimiter, cfg.header)

    good_records: List[Dict] = []
    error_records: List[Dict] = []

    expected_len = len(header)

    for row_index, row in enumerate(rows, start=2 if cfg.header else 1):
        # Malformed row length
        if len(row) != expected_len:
            error_records.append({
                "partner": partner_key,
                "file": input_file,
                "row_number": row_index,
                "error": f"malformed_row_length expected={expected_len} got={len(row)}"
            })
            if drop_invalid:
                continue

        standard = map_to_standard(header, row, cfg)
        standard = apply_transformations(standard)

        errors = validate(standard)
        if errors:
            error_records.append({
                "partner": partner_key,
                "file": input_file,
                "row_number": row_index,
                "error": ",".join(errors)
            })
            if drop_invalid:
                continue

        good_records.append(standard)

    return good_records, error_records


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    """Writes output CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in fieldnames})


def parse_inputs(pairs: List[str]) -> Dict[str, str]:
    """
    Parses --inputs partner=filepath partner=filepath
    Example:
      --inputs acme=data/acme.txt bettercare=data/bettercare.csv
    """
    result: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid input '{item}'. Expected format partner_key=path")
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--inputs", required=True, nargs="+", help="partner_key=filepath pairs")
    parser.add_argument("--output", required=True, help="Unified output CSV path")
    parser.add_argument("--drop-invalid", action="store_true", help="Drop invalid rows instead of keeping them")

    args = parser.parse_args()

    configs = load_configs(args.config)
    inputs = parse_inputs(args.inputs)

    all_good: List[Dict] = []
    all_errors: List[Dict] = []

    for partner_key, input_file in inputs.items():
        if partner_key not in configs:
            raise KeyError(f"Partner '{partner_key}' not found in config. Available: {list(configs.keys())}")

        good, errors = process_partner(partner_key, input_file, configs[partner_key], args.drop_invalid)
        all_good.extend(good)
        all_errors.extend(errors)

    # Write unified output
    write_csv(args.output, STANDARD_FIELDS, all_good)

    # Write error report next to output
    error_path = str(Path(args.output).with_suffix("")) + "_errors.csv"
    if all_errors:
        write_csv(error_path, ["partner", "file", "row_number", "error"], all_errors)

    print(f"✅ Unified dataset written to: {args.output} (rows={len(all_good)})")
    if all_errors:
        print(f"⚠️  Error report written to: {error_path} (errors={len(all_errors)})")


if __name__ == "__main__":
    main()

