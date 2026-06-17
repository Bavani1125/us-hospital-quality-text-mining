import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load data
df = pd.read_csv('/mnt/data/Survey Data Median.csv')

# Renaming for short labels
rename_map = {
 'Patients who reported that staff definitely gave care in a professional way and the facility was clean': 'ProfClean_Def',
 'Patients who reported that staff somewhat gave care in a professional way or the facility was somewhat clean': 'ProfClean_Some',
 'Patients who reported that staff did not give care in a professional way or the facility was not clean': 'ProfClean_No',
 'Patients who reported that staff definitely communicated about what to expect during and after the procedure': 'Comm_Def',
 'Patients who reported that staff somewhat communicated about what to expect during and after the procedure': 'Comm_Some',
 'Patients who reported that staff did not communicate about what to expect during and after the procedure': 'Comm_No',
 'Patients who gave the facility a rating of 9 or 10 on a scale from 0 (lowest) to 10 (highest)': 'Rate_9_10',
 'Patients who gave the facility a rating of 7 or 8 on a scale from 0 (lowest) to 10 (highest)': 'Rate_7_8',
 'Patients who gave the facility a rating of 0 to 6 on a scale from 0 (lowest) to 10 (highest)': 'Rate_0_6',
 'Patients who reported YES they would DEFINITELY recommend the facility to family or friends': 'Recm_Def',
 'Patients who reported PROBABLY YES they would recommend the facility to family or friends': 'Recm_Prob',
 'Patients who reported NO, they would not recommend the facility to family or friends': 'Recm_No'
}

df = df.rename(columns=rename_map)

# Numeric columns
numeric_df = df.select_dtypes(include=[np.number])
cols = numeric_df.columns
n = len(cols)

# Output file
pdf_path = "/mnt/data/scatter_matrix_trendlines_v2.pdf"

with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n))

    for i, col_y in enumerate(cols):
        for j, col_x in enumerate(cols):
            ax = axes[i, j]
            ax.scatter(numeric_df[col_x], numeric_df[col_y], s=10)

            # Curved trendline
            try:
                z = np.polyfit(numeric_df[col_x], numeric_df[col_y], 2)
                p = np.poly1d(z)
                xs = np.linspace(numeric_df[col_x].min(), numeric_df[col_x].max(), 200)
                ax.plot(xs, p(xs))
            except:
                pass

            # Labeling
            if i == n-1:
                ax.set_xlabel(col_x, fontsize=8, rotation=45)
            if j == 0:
                ax.set_ylabel(col_y, fontsize=8)

            ax.tick_params(axis='both', labelsize=6)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

pdf_path
