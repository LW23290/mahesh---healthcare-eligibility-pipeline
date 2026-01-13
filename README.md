
# mahesh---healthcare-eligibility-pipeline
# Healthcare Eligibility Pipeline

This project implements a **configuration-driven data ingestion pipeline** for healthcare eligibility files.  
It ingests partner files with different formats and column names, standardizes them into a single schema, applies data quality rules, and produces one unified dataset for downstream systems.

---

## What This Pipeline Does

For each partner file, the pipeline:

1. Reads the file using partner-specific settings (delimiter, header) from YAML config  
2. Maps partner columns → a standard schema  
3. Applies required transformations:
   - `external_id` mapped from partner’s unique ID
   - `first_name`, `last_name` → Title Case
   - `dob` → ISO-8601 (`YYYY-MM-DD`)
   - `email` → lowercase
   - `phone` → `XXX-XXX-XXXX`
   - `partner_code` → hardcoded per partner
4. Validates and handles bad data:
   - `external_id` is required
   - Malformed rows are logged (job does not crash)
   - Invalid DOB formats are flagged
5. Writes:
   - A unified output dataset
   - An error report for onboarding/debugging

---

## Standard Output Schema

| Field         | Description |
|---------------|-------------|
| external_id   | Partner’s unique member ID |
| first_name    | Title-cased |
| last_name     | Title-cased |
| dob           | ISO-8601 date (`YYYY-MM-DD`) |
| email         | Lowercase |
| phone         | `XXX-XXX-XXXX` |
| partner_code  | Partner identifier |
| is_valid      | Row-level validity flag |
| error_reason  | Reason(s) a row is invalid |

---

## How to Run

```bash
python eligibility_pipeline.py \
  --config configs/partners.yaml \
  --inputs acme=data/acme.txt bettercare=data/bettercare.csv \
  --output output/unified_eligibility.csv \
  --drop-invalid

