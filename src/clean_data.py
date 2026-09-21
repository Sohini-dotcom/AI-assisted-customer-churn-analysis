"""
clean_data.py
Cleans the raw Telco churn file and saves a 2000-row, 12-column sample.

- READS   data/raw/Telco-Customer-Churn.csv   (never modified)
- WRITES  data/processed/telco_churn_clean_2000.csv
- WRITES  outputs/cleaning_report.txt
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

# Step 0: Set up the file paths (relative to the project folder, so the
# script works no matter which folder you run it from).
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_PATH = PROJECT_DIR / "data" / "processed" / "telco_churn_clean_2000.csv"
REPORT_PATH = PROJECT_DIR / "outputs" / "cleaning_report.txt"

# The exact 12 columns we keep, in this order (from CLAUDE.md)
KEEP_COLUMNS = [
    "customerID", "SeniorCitizen", "Dependents", "tenure", "InternetService",
    "OnlineSecurity", "TechSupport", "Contract", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
SAMPLE_SIZE = 2000
RANDOM_STATE = 42

# Everything we print is also saved here, so it can go into the report file
report_lines = []


def log(text=""):
    """Print a line to the screen AND remember it for the report file."""
    print(text)
    report_lines.append(text)


def section(title):
    """Print a clear heading."""
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def stop_with_error(message):
    """Stop the script immediately with a clear error message."""
    sys.exit(f"\nSTOPPED - {message}")


def check(condition, message):
    """If a check fails, stop the script. Otherwise carry on quietly."""
    if not condition:
        stop_with_error(message)


def sha256_of_file(path):
    """Return the SHA-256 hash (a 'fingerprint') of a file.
    If even one byte of the file changes, the hash changes completely."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:  # "rb" = read only, as raw bytes
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# Step 1: Fingerprint the raw file BEFORE we touch anything.
# We compare this to a second fingerprint at the very end.
hash_before = sha256_of_file(RAW_PATH)

section("CLEANING REPORT - Telco Customer Churn")
log(f"Raw file: {RAW_PATH.relative_to(PROJECT_DIR)}")
log(f"SHA-256 before: {hash_before}")

# Step 2: Load the raw CSV (read only) into a table.
raw = pd.read_csv(RAW_PATH)
raw_rows, raw_cols = raw.shape

# Remember some "before" facts so we can show before/after at the end
raw_blank_total = int((raw["TotalCharges"].str.strip() == "").sum())
raw_total_dtype = str(raw["TotalCharges"].dtype)
raw_churn_rate = (raw["Churn"] == "Yes").mean()

# Step 3: Keep only the 12 columns we need.
# .copy() makes a separate table, so the loaded data stays as it was.
missing_cols = [c for c in KEEP_COLUMNS if c not in raw.columns]
check(not missing_cols, f"These expected columns are not in the raw file: {missing_cols}")
df = raw[KEEP_COLUMNS].copy()

# Step 4: Fix TotalCharges.
# Blank TotalCharges is only acceptable when tenure is 0 (a brand-new
# customer who has not been billed yet). Those become 0.0.
# Any OTHER problem value stops the script with an error.
is_blank = df["TotalCharges"].str.strip() == ""
fix_mask = is_blank & (df["tenure"] == 0)
blanks_fixed_full = int(fix_mask.sum())
df.loc[fix_mask, "TotalCharges"] = "0.0"

# Turn the text into numbers. Anything that cannot be converted becomes NaN,
# which we then catch below.
total_as_number = pd.to_numeric(df["TotalCharges"], errors="coerce")
bad_rows = total_as_number.isna()
if bad_rows.any():
    problem = df.loc[bad_rows, ["customerID", "tenure", "TotalCharges"]].head(10)
    stop_with_error(
        f"{int(bad_rows.sum())} TotalCharges value(s) could not be converted to a "
        f"number, and they are NOT blank rows with tenure 0. First few:\n{problem}"
    )
df["TotalCharges"] = total_as_number

# Step 5: Check the cleaned (full) data for problems.
# Any failure here stops the script.
check(df.isna().sum().sum() == 0, "There are missing values after cleaning.")
check(df.duplicated().sum() == 0, "There are duplicate rows.")
check(df["customerID"].is_unique, "customerID is not unique.")
check(df["TotalCharges"].dtype == "float64", "TotalCharges is not a number column.")
text_cols = df.select_dtypes(exclude="number").columns
blank_text = sum(int((df[c].str.strip() == "").sum()) for c in text_cols)
check(blank_text == 0, f"{blank_text} blank text values remain.")

section("STEP A - Cleaning the full dataset")
log(f"Kept {len(KEEP_COLUMNS)} of {raw_cols} columns.")
log(f"TotalCharges blanks set to 0.0 (tenure was 0): {blanks_fixed_full}")
log("TotalCharges converted from text to a number (float64).")
log("Checks passed: no missing values, no blank text, no duplicate rows, "
    "customerID unique.")

# Step 6: Take the stratified random sample of 2000 rows.
# "Stratified" = each Churn group (Yes/No) is sampled in the same proportion
# as in the full data, so the sample keeps the same churn rate.
# We work out exactly how many rows to take from each group so the total is
# exactly 2000 (the largest group absorbs any rounding difference).
group_sizes = df["Churn"].value_counts()
rows_per_group = (group_sizes / len(df) * SAMPLE_SIZE).round().astype(int)
rows_per_group[group_sizes.idxmax()] += SAMPLE_SIZE - rows_per_group.sum()

parts = [
    df[df["Churn"] == group].sample(n=n, random_state=RANDOM_STATE)
    for group, n in rows_per_group.items()
]
# Shuffle so Yes/No rows are mixed together, not stacked in two blocks
sample = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)
sample = sample.reset_index(drop=True)

# Step 7: Validate the sample. Any failure here stops the script.
check(len(sample) == SAMPLE_SIZE, f"Sample has {len(sample)} rows, expected {SAMPLE_SIZE}.")
check(list(sample.columns) == KEEP_COLUMNS, "Sample columns are not the expected 12.")
check(sample.isna().sum().sum() == 0, "Sample contains missing values.")
check(sample["customerID"].is_unique, "customerID is not unique in the sample.")
sample_churn_rate = (sample["Churn"] == "Yes").mean()
full_churn_rate = (df["Churn"] == "Yes").mean()
check(abs(sample_churn_rate - full_churn_rate) < 0.001,
      "Sample churn rate differs from the full data by more than 0.1 points.")

section("STEP B - Stratified sample")
log(f"Sample size: {len(sample)} rows x {sample.shape[1]} columns "
    f"(random_state={RANDOM_STATE})")
log("Rows taken per Churn group: "
    + ", ".join(f"{g}={n}" for g, n in rows_per_group.items()))
log(f"Churn rate  full data: {full_churn_rate:.2%}   sample: {sample_churn_rate:.2%}")
log(f"Rows in the sample with TotalCharges = 0.0 (new customers): "
    f"{int((sample['tenure'] == 0).sum())}")

# Step 8: Compare category mix, full data vs sample (for information only).
# Small differences are normal; we report the biggest one for each column.
log("\nCategory mix, full data vs sample (largest gap in percentage points):")
for col in ["Contract", "InternetService", "PaymentMethod"]:
    full_pct = df[col].value_counts(normalize=True) * 100
    sample_pct = sample[col].value_counts(normalize=True) * 100
    gap = (full_pct - sample_pct.reindex(full_pct.index).fillna(0)).abs().max()
    log(f"  {col:<16} largest gap: {gap:.2f} points")

# Step 9: Save the cleaned sample (index=False stops pandas writing row numbers).
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(PROCESSED_PATH, index=False)

# Step 10: Read the saved file back in and confirm it is what we expect.
saved = pd.read_csv(PROCESSED_PATH)
check(saved.shape == (SAMPLE_SIZE, len(KEEP_COLUMNS)), "Saved file has the wrong shape.")
check(saved.isna().sum().sum() == 0, "Saved file contains missing values.")
check(saved["TotalCharges"].dtype == "float64", "Saved TotalCharges is not numeric.")
log(f"\nSaved: {PROCESSED_PATH.relative_to(PROJECT_DIR)} "
    f"({saved.shape[0]} rows x {saved.shape[1]} columns, re-read and verified)")

# Step 11: Fingerprint the raw file again and compare with the first one.
hash_after = sha256_of_file(RAW_PATH)
raw_unchanged = hash_before == hash_after

# Step 12: Print the before/after summary.
section("BEFORE / AFTER SUMMARY")
log(f"{'':<34}{'BEFORE (raw)':<18}{'AFTER (clean sample)'}")
log(f"{'Rows':<34}{raw_rows:<18}{len(sample)}")
log(f"{'Columns':<34}{raw_cols:<18}{sample.shape[1]}")
log(f"{'TotalCharges data type':<34}{raw_total_dtype:<18}{sample['TotalCharges'].dtype}")
log(f"{'Blank TotalCharges values':<34}{raw_blank_total:<18}"
    f"{int((sample['TotalCharges'].isna()).sum())}")
log(f"{'Missing values (all columns)':<34}{int(raw.isna().sum().sum()):<18}"
    f"{int(sample.isna().sum().sum())}")
log(f"{'Churn rate':<34}{raw_churn_rate:<18.2%}{sample_churn_rate:.2%}")

section("RAW FILE SAFETY CHECK (SHA-256)")
log(f"Before: {hash_before}")
log(f"After:  {hash_after}")
log("Result: " + ("IDENTICAL - the raw file was not modified."
                  if raw_unchanged else "DIFFERENT - THE RAW FILE CHANGED!"))

# Step 13: Save the report as a text file.
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print(f"\nReport saved: {REPORT_PATH.relative_to(PROJECT_DIR)}")

# Step 14: If the raw file changed, shout about it (this should never happen).
check(raw_unchanged, "The raw file's SHA-256 hash changed. Investigate immediately.")
