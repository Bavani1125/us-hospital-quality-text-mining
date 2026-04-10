# survey_correlation_B_to_M.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import numpy as np
import string

# Step 1: Load Excel file
file_path = "Survey Data Median.xlsx"
df = pd.read_excel(file_path)

# Step 2: Drop non-numeric columns (exclude 'State' or first column)
# If first column is 'State', drop it
if df.columns[0].lower() in ['state', 'states']:
    df_numeric = df.drop(df.columns[0], axis=1)
else:
    df_numeric = df.select_dtypes(include='number')

# Step 3: Determine how many numeric columns we have
cols = df_numeric.columns
num_cols = len(cols)

# Step 4: Create alphabet labels starting from 'B' (so first label = B, not A)
# Example: B, C, D, E, ...
letters = list(string.ascii_uppercase[1:1 + num_cols])  # skip 'A' (State column)

# Step 5: Create a mapping for reference
col_map = pd.DataFrame({'Letter': letters, 'Original Column': cols})
map_output = os.path.join(os.path.dirname(file_path), "column_mapping_B_to_M.xlsx")
col_map.to_excel(map_output, index=False)

# Step 6: Rename numeric columns to letter labels
df_numeric.columns = letters

# Step 7: Compute R (correlation) and P (significance) matrices
r_matrix = pd.DataFrame(np.zeros((num_cols, num_cols)), columns=letters, index=letters)
p_matrix = pd.DataFrame(np.zeros((num_cols, num_cols)), columns=letters, index=letters)

for col1 in letters:
    for col2 in letters:
        r, p = pearsonr(df_numeric[col1], df_numeric[col2])
        r_matrix.loc[col1, col2] = r
        p_matrix.loc[col1, col2] = p

# Step 8: Save matrices
r_output = os.path.join(os.path.dirname(file_path), "survey_r_values_B_to_M.xlsx")
p_output = os.path.join(os.path.dirname(file_path), "survey_p_values_B_to_M.xlsx")
r_matrix.to_excel(r_output)
p_matrix.to_excel(p_output)

# Step 9: Plot correlogram (R values)
plt.figure(figsize=(9, 7))
sns.heatmap(r_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
plt.title("Survey Data Correlation (Columns B–M)", fontsize=14)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

output_img = os.path.join(os.path.dirname(file_path), "survey_correlogram_B_to_M.png")
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Column mapping saved to: {map_output}")
print(f"✅ R-values saved to: {r_output}")
print(f"✅ P-values saved to: {p_output}")
print(f"✅ Correlogram saved to: {output_img}")
