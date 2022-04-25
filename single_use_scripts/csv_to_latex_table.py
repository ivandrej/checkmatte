import pandas as pd

path = "/media/andivanov/DATA/report/training_higher_resolutions/res512x288_reduced_vs_rvm_reduced .csv"

df = pd.read_csv(path)

print("\\hline")
row_str = f"\multirow{{3}}{{4em}}{{10 frames}} & 0 & {df.loc[0, 'pha_mad']:.4f} & {df.loc[0, 'pha_bgr_mad']:.4f} \\\ " \
          f"& (5, 10) & {df.loc[2, 'pha_mad']:.4f} & {df.loc[2, 'pha_bgr_mad']:.4f} \\\ " \
          f"& (25, 30) & {df.loc[1, 'pha_mad']:.4f} & {df.loc[1, 'pha_bgr_mad']:.4f} \\\ "
print(row_str)
print("\\hline")
row_str = f"\multirow{{3}}{{4em}}{{30 frames}} & 0 & {df.loc[3, 'pha_mad']:.4f} & {df.loc[3, 'pha_bgr_mad']:.4f} \\\ " \
          f"& (5, 10) & {df.loc[5, 'pha_mad']:.4f} & {df.loc[5, 'pha_bgr_mad']:.4f} \\\ " \
          f"& (25, 30) & {df.loc[4, 'pha_mad']:.4f} & {df.loc[4, 'pha_bgr_mad']:.4f} \\\ "
print(row_str)
print("\\hline")
row_str = f"\multirow{{3}}{{4em}}{{50 frames}} & 0 & {df.loc[6, 'pha_mad']:.4f} & {df.loc[6, 'pha_bgr_mad']:.4f} \\\ " \
          f"& (5, 10) & {df.loc[8, 'pha_mad']:.4f} & {df.loc[8, 'pha_bgr_mad']:.4f} \\\ " \
          f"& (25, 30) & {df.loc[7, 'pha_mad']:.4f} & {df.loc[7, 'pha_bgr_mad']:.4f} \\\ "
print(row_str)
print("\\hline")
row_str = f"\multirow{{3}}{{4em}}{{80 frames}} & 0 & {df.loc[9, 'pha_mad']:.4f} & {df.loc[9, 'pha_bgr_mad']:.4f} \\\ " \
          f"& (5, 10) & {df.loc[11, 'pha_mad']:.4f} & {df.loc[11, 'pha_bgr_mad']:.4f} \\\ " \
          f"& (25, 30) & {df.loc[10, 'pha_mad']:.4f} & {df.loc[10, 'pha_bgr_mad']:.4f} \\\ "
print(row_str)
print("\\hline")
print("\\hline")
row_str = f"RVM & RVM & {df.loc[12, 'pha_mad']:.4f} & {df.loc[12, 'pha_bgr_mad']:.4f} \\\ "
print(row_str)
print("\\hline")
print(df)

