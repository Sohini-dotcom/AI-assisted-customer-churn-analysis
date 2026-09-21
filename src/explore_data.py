"""
explore_data.py
Read-only exploration of the raw Telco churn file.
This script only READS the CSV. It never writes or changes any data.
"""

import pandas as pd

# Path to the raw file (we only read it, never write to it)
RAW_PATH = "data/raw/Telco-Customer-Churn.csv"


def section(title):
    """Print a clear heading so the output is easy to scan."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# Step 1: Load the CSV into a table (a "DataFrame").
# We do NOT let pandas guess "missing" values beyond its defaults, so that
# blank strings stay visible as text and we can count them ourselves.
df = pd.read_csv(RAW_PATH)

# Step 2: Number of rows and columns, and each column's data type
section("1. SHAPE AND DATA TYPES")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nData type of each column:")
print(df.dtypes)

# Step 3: Find the text columns (anything that is not a number).
# We use this list several times below.
text_cols = df.select_dtypes(exclude="number").columns.tolist()

# Step 4: Missing values per column (real NaN / empty cells)
section("2. MISSING VALUES (NaN) PER COLUMN")
print(df.isna().sum())

# Step 5: Blank or whitespace-only text values per column.
# These are NOT counted as missing by pandas, so we check them separately.
# Example: " " (a single space) looks like data but is really empty.
section("3. BLANK OR WHITESPACE-ONLY TEXT VALUES PER COLUMN")
blank_counts = {}
for col in text_cols:
    # .str.strip() removes spaces; if nothing is left, the value was blank
    blank_counts[col] = int((df[col].str.strip() == "").sum())
print(pd.Series(blank_counts))

# Step 6: Duplicate rows, and is customerID unique?
section("4. DUPLICATES")
print(f"Fully duplicated rows: {df.duplicated().sum()}")
print(f"Duplicated customerID values: {df['customerID'].duplicated().sum()}")
print(f"Is customerID unique? {df['customerID'].is_unique}")

# Step 7: Basic statistics for the numeric columns
section("5. BASIC STATISTICS (NUMERIC COLUMNS)")
print(df.describe().T)

# Step 8: Count of each category in every text column.
# We skip customerID (every value is different) and TotalCharges
# (it is really a number stored as text, so it has thousands of values).
section("6. CATEGORY COUNTS FOR TEXT COLUMNS")
skip = ["customerID", "TotalCharges"]
for col in text_cols:
    if col in skip:
        continue
    print(f"\n--- {col} ---")
    # dropna=False so any missing values would also show up in the counts
    print(df[col].value_counts(dropna=False))

# Step 9: Churn rate overall, and by Contract, InternetService, PaymentMethod.
# We turn "Yes"/"No" into 1/0 so that the average = the churn rate.
section("7. CHURN RATE")
churned = (df["Churn"] == "Yes").astype(int)
print(f"Overall churn rate: {churned.mean():.1%}")
for col in ["Contract", "InternetService", "PaymentMethod"]:
    print(f"\nChurn rate by {col}:")
    summary = churned.groupby(df[col]).agg(customers="count", churn_rate="mean")
    summary["churn_rate"] = (summary["churn_rate"] * 100).round(1).astype(str) + "%"
    print(summary)

# Step 10: Rows where TotalCharges is blank, and what their tenure is
section("8. ROWS WITH BLANK TotalCharges")
blank_total = df["TotalCharges"].str.strip() == ""
print(f"Number of rows with blank TotalCharges: {blank_total.sum()}")
print("\nTenure values of those rows:")
print(df.loc[blank_total, "tenure"].value_counts())
print("\nThose rows (selected columns):")
print(df.loc[blank_total, ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

# Step 11: Is TotalCharges roughly tenure x MonthlyCharges?
# First convert TotalCharges to a number. Blanks become NaN (we only do this
# in memory for the check; the file is untouched).
section("9. IS TotalCharges ROUGHLY tenure x MonthlyCharges?")
total_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
expected = df["tenure"] * df["MonthlyCharges"]

# Only compare rows that have a real TotalCharges and tenure above 0
ok = total_num.notna() & (df["tenure"] > 0)
ratio = total_num[ok] / expected[ok]
diff = total_num[ok] - expected[ok]

print(f"Rows compared: {ok.sum()}")
print(f"Correlation between TotalCharges and tenure x MonthlyCharges: "
      f"{total_num[ok].corr(expected[ok]):.4f}")
print("\nRatio  TotalCharges / (tenure x MonthlyCharges)  (1.0 = exact match):")
print(ratio.describe())
print(f"\nRows within 10% of the expected value: {((ratio - 1).abs() <= 0.10).mean():.1%}")
print(f"Rows more than 25% away from the expected value: {((ratio - 1).abs() > 0.25).sum()}")
print("\nDifference (TotalCharges - expected), in dollars:")
print(diff.describe())

# Step 12: Confirm the file was only read
section("DONE")
print("This script only read the raw file. Nothing was changed or saved.")
