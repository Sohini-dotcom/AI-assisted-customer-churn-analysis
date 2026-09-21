"""
features.py
Adds 6 new "feature" columns to the cleaned customer data.

- READS   data/processed/telco_churn_clean_2000.csv   (never modified)
- WRITES  data/processed/customers_features.csv       (12 original + 6 new columns)
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

# Step 0: Set up the file paths (relative to the project folder, so the
# script works no matter which folder you run it from).
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "data" / "processed" / "telco_churn_clean_2000.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "customers_features.csv"

# The 6 new columns, in the order they will appear at the end of the table
NEW_COLUMNS = [
    "churn_flag", "senior_label", "tenure_group",
    "monthly_charges_band", "has_internet", "auto_pay",
]

# Bucket edges and labels (approved). Each bucket INCLUDES its upper edge,
# so tenure 12 is in "0-12 months" and a $70.00 bill is in "2-Medium".
TENURE_EDGES = [-1, 12, 24, 48, 72]          # -1 so that tenure 0 is included
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
CHARGE_EDGES = [0, 35, 70, 90, float("inf")]  # inf = "no upper limit"
CHARGE_LABELS = [
    "1-Low (up to $35)", "2-Medium ($35-$70)",
    "3-High ($70-$90)", "4-Very high (over $90)",
]


def stop_with_error(message):
    """Stop the script immediately with a clear error message."""
    sys.exit(f"\nSTOPPED - {message}")


def check(condition, message):
    """If a safety check fails, stop the script. Otherwise carry on quietly."""
    if not condition:
        stop_with_error(message)


def sha256_of_file(path):
    """Return the SHA-256 'fingerprint' of a file.
    If even one byte of the file changes, the fingerprint changes."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:  # "rb" = read only
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def section(title):
    """Print a clear heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# Step 1: Fingerprint the input file BEFORE we do anything, so we can prove
# at the end that we did not change it.
check(INPUT_PATH.exists(), f"Input file not found: {INPUT_PATH}")
hash_before = sha256_of_file(INPUT_PATH)

# Step 2: Load the cleaned data (read only) and check it is what we expect.
source = pd.read_csv(INPUT_PATH)
check(source.shape == (2000, 12), f"Input should be 2000 x 12, got {source.shape}.")
original_columns = list(source.columns)

# Step 3: Work on a COPY, so the loaded table stays exactly as it was.
# This lets us compare the two at the end.
df = source.copy()

# Step 4: churn_flag = 1 if the customer churned ("Yes"), otherwise 0.
# Numbers are easier than words for averages (the average IS the churn rate)
# and for machine-learning models later.
df["churn_flag"] = (df["Churn"] == "Yes").astype(int)

# Step 5: senior_label = "Yes" if SeniorCitizen is 1, otherwise "No".
# This turns the 0/1 code into readable words for the dashboard.
df["senior_label"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

# Step 6: tenure_group = put each customer's tenure (months) into a bucket.
# pd.cut() sorts numbers into buckets using the edges defined at the top.
# If a value falls outside every bucket it becomes empty (NaN); a safety
# check below would then stop the script.
df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_EDGES, labels=TENURE_LABELS)

# Step 7: monthly_charges_band = same idea, using MonthlyCharges (dollars).
# The number at the start of each label makes Power BI sort them correctly.
df["monthly_charges_band"] = pd.cut(
    df["MonthlyCharges"], bins=CHARGE_EDGES, labels=CHARGE_LABELS
)

# Step 8: Check the two bucket columns for empty values BEFORE turning them
# into plain text (otherwise an empty value would silently become "nan").
for col in ["tenure_group", "monthly_charges_band"]:
    check(df[col].notna().all(), f"{col} has values that fit no bucket.")
    df[col] = df[col].astype(str)  # plain text, easier to save and use

# Step 9: has_internet = "Yes" if InternetService is anything except "No".
df["has_internet"] = (df["InternetService"] != "No").map({True: "Yes", False: "No"})

# Step 10: auto_pay = "Yes" if PaymentMethod contains the word "automatic"
# (that is "Bank transfer (automatic)" and "Credit card (automatic)").
df["auto_pay"] = df["PaymentMethod"].str.contains("automatic").map(
    {True: "Yes", False: "No"}
)

# Step 11: Put the columns in a clear order: 12 originals, then the 6 new ones.
df = df[original_columns + NEW_COLUMNS]

# Step 12: Safety checks on the table before saving. Any failure stops the script.
check(len(df) == 2000, f"Expected 2000 rows, got {len(df)}.")
check(df.shape[1] == 18, f"Expected exactly 18 columns, got {df.shape[1]}.")
check(df[original_columns].equals(source), "The 12 original columns were changed.")
for col in NEW_COLUMNS:
    check(df[col].notna().all(), f"New column {col} has missing values.")
    if df[col].dtype != "int64":  # text columns: also look for blank text
        check((df[col].str.strip() != "").all(), f"New column {col} has blank values.")
check(set(df["churn_flag"]) <= {0, 1}, "churn_flag contains something other than 0/1.")

# Step 13: Save the new file (index=False stops pandas writing row numbers).
df.to_csv(OUTPUT_PATH, index=False)

# Step 14: Read the saved file back in and confirm it is really what we expect.
saved = pd.read_csv(OUTPUT_PATH)
check(saved.shape == (2000, 18), f"Saved file should be 2000 x 18, got {saved.shape}.")
check(list(saved.columns) == original_columns + NEW_COLUMNS, "Saved columns are wrong.")
check(saved[original_columns].equals(source), "Saved original columns differ from the input.")
check(saved[NEW_COLUMNS].notna().all().all(), "Saved new columns contain empty values.")

# Step 15: Fingerprint the input file again and compare with Step 1.
hash_after = sha256_of_file(INPUT_PATH)

section("SAFETY CHECKS")
print(f"Rows: {saved.shape[0]}   Columns: {saved.shape[1]}")
print("12 original columns unchanged:   PASSED")
print("No blank values in new columns:  PASSED")
print(f"Saved: {OUTPUT_PATH.relative_to(PROJECT_DIR)}")
print(f"Input file SHA-256 before: {hash_before}")
print(f"Input file SHA-256 after:  {hash_after}")
print("Input file:", "IDENTICAL - not modified." if hash_before == hash_after
      else "DIFFERENT - THE INPUT FILE CHANGED!")
check(hash_before == hash_after, "The input file's SHA-256 hash changed.")

# Step 16: For every new column, show how many customers are in each group
# and the churn rate of each group (churn rate = average of churn_flag).
# sort_index() puts the groups in their natural order (the labels are
# written so that alphabetical order is also the correct order).
overall_rate = saved["churn_flag"].mean()
section(f"NEW COLUMNS: COUNTS AND CHURN RATE   (overall churn rate: {overall_rate:.1%})")

for col in NEW_COLUMNS:
    print(f"\n--- {col} ---")
    if col == "churn_flag":
        # Grouping churn_flag by itself would just show 0% and 100%, so we
        # show the counts only.
        print(saved[col].value_counts().sort_index().to_string())
        continue
    summary = saved.groupby(col)["churn_flag"].agg(
        customers="count", churned="sum", churn_rate="mean"
    ).sort_index()
    summary["churn_rate"] = (summary["churn_rate"] * 100).round(1).astype(str) + "%"
    print(summary.to_string())
