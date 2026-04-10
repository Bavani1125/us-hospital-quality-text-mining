# correlation_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import numpy as np

# Step 1: Load Excel file
file_path = "hospital general info medians.xlsx"
df = pd.read_excel(file_path)

# Step 2: Keep only numeric columns (exclude 'State')
if 'State' in df.columns:
    df_numeric = df.drop(columns=['State'])
else:
    df_numeric = df.select_dtypes(include='number')

cols = df_numeric.columns

# Step 3: Compute correlation (R) and p-value matrices
r_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
p_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)

for col1 in cols:
    for col2 in cols:
        r, p = pearsonr(df_numeric[col1], df_numeric[col2])
        r_matrix.loc[col1, col2] = r
        p_matrix.loc[col1, col2] = p

# Step 4: Save both matrices as Excel files
r_output = os.path.join(os.path.dirname(file_path), "correlation_r_values.xlsx")
p_output = os.path.join(os.path.dirname(file_path), "correlation_p_values.xlsx")
r_matrix.to_excel(r_output)
p_matrix.to_excel(p_output)

# Step 5: Plot and save correlogram (based on R values)
plt.figure(figsize=(10, 8))
sns.heatmap(r_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix (Correlogram) - R Values", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

output_img = os.path.join(os.path.dirname(file_path), "correlogram.png")
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Correlation R-values saved to: {r_output}")
print(f"✅ Correlation P-values saved to: {p_output}")
print(f"✅ Correlogram image saved to: {output_img}")
