import pandas as pd
import os

dir = "20240729_113758_ks_fno"
# metric = 'EnergyScoreValidation'

path = os.path.join("results", dir)

files = os.listdir(path)
files.sort()  # alphabetic order, since this is the order corresponding to the entries in test.csv
results = pd.read_csv(os.path.join(path, "test.csv"), index_col=0).T


indices_best = results.groupby(
    "uncertainty_quantification"
).EnergyScoreValidation.idxmin()

results[
    [
        "uncertainty_quantification",
        "MSETest",
        "EnergyScoreTest",
        "CoverageTest",
        "IntervalWidthTest"
    ]
].loc[indices_best].to_csv(os.path.join(path, "best_models.csv"), index=False)
