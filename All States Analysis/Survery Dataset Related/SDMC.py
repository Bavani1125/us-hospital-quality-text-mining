import pandas as pd
import numpy as np

# =========================
# CONFIGURATION
# =========================
INPUT_FILE = "SDMCAL.xlsx"  # or .csv
OUTPUT_FILE = "statewise_median.xlsx"
STATE_COLUMN = "State"

# =========================
# LOAD DATA
# =========================
if INPUT_FILE.endswith(".csv"):
    df = pd.read_csv(INPUT_FILE)
else:
    df = pd.read_excel(INPUT_FILE)

# =========================
# HANDLE 'Not Available'
# =========================
# Replace text-based missing values with NaN
df = df.replace(
    ["Not Available", "Not available", "NA", "N/A", ""],
    np.nan
)

# =========================
# CONVERT NUMERIC COLUMNS
# =========================
# Force numeric conversion where possible
for col in df.columns:
    if col != STATE_COLUMN:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# SELECT NUMERIC COLUMNS
# =========================
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Optional: exclude ID columns
numeric_cols = [col for col in numeric_cols if col != "Facility ID"]

# =========================
# STATE-WISE MEDIAN
# =========================
state_median_df = (
    df
    .groupby(STATE_COLUMN)[numeric_cols]
    .median()
    .reset_index()
)

# =========================
# SAVE OUTPUT
# =========================
state_median_df.to_excel(OUTPUT_FILE, index=False)

print("✅ State-wise median computed (missing values ignored).")
print(f"📄 Output saved to: {OUTPUT_FILE}")
