import pandas as pd
import os

def get_folders(dir: str = "") -> list:
    folders = os.listdir(f"{dir}/")
    return folders

def get_metrics():
    metrics = ['CRPSValidation', 'Gaussian NLLValidation',
       'CoverageValidation', 'IntervalWidthValidation', 'MSETest',
       'EnergyScoreTest', 'CRPSTest', 'Gaussian NLLTest', 'CoverageTest',
       'IntervalWidthTest']
    return metrics

def extract_name(folder_name):
    parts = folder_name.split('_')
    if parts[-2] == "sr":
        return parts[-2] + '_' + parts[-1]  # Extract the last two parts (e.g., 'sr_reparam')
    else:
        return parts[-1] 

def get_methods(results_dir:str, experiment:str, model:str) -> list:
    base_path = f"results/{results_dir}/{experiment}/{model}"
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    folders.sort()
    methods = [extract_name(f) for f in folders]
    return methods

def process_experiment(results_dir: str, experiment: str, model: str) -> pd.DataFrame:
    path = f"{results_dir}/{experiment}/{model}/"
    results = pd.DataFrame()
    # Loop over subfolders
    folders = os.listdir(path)
    for sf in folders:
        if os.path.isdir(path + sf):
            file = os.path.join(path+sf, "test.csv")
            # Read file if exits
            if os.path.exists(file):
                results_df = pd.read_csv(file, index_col=0)
                results = pd.concat([results, results_df], axis = 1)
    metrics = get_metrics()
    rows = metrics.copy()
    rows.append("uncertainty_quantification")
    results = results.loc[rows]
    results.loc[metrics] = results.loc[metrics].astype("float32")
    results = results.transpose()
    # Group by uncertainty quantification method
    mean = results.groupby("uncertainty_quantification").mean().astype("float32")
    mean.insert(0, "Statistic", "Mean")
    std = results.groupby("uncertainty_quantification").std().astype("float32")
    std.insert(0, "Statistic", "Std")

    results_df = pd.concat([mean.transpose(), std.transpose()], axis = 1)
    results_df = results_df[results_df.columns.sort_values().unique()]
    return results_df

def create_latex_table(results_df: pd.DataFrame, results_dir:str, experiment: str, model: str) -> str:
    # Initialize an empty DataFrame to store the formatted values
    formatted_df = pd.DataFrame()
    metrics = results_df.index[-6:] # Extract only test metrics
    methods = list(results_df.columns.unique())  # Methods are the top level of the columns MultiIndex

    # Create a new DataFrame with the method as the index and metrics as columns
    for metric in metrics:
        formatted_df[metric] = [
            f"\\makecell{{{results_df.loc[metric, method].values[0]:.4f} \\\\ ($\\pm$ {results_df.loc[metric, method].values[1]:.4f})}}"
            for method in methods
        ]
    formatted_df.index = methods
    latex_table = formatted_df.to_latex(escape=False)
    #  save to a file
    with open(f"{results_dir}/{experiment}/{model}/aggregated_results.tex", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    results_dir = "results/optimal_hp_multiple_seeds"
    experiments = get_folders(dir = results_dir)
    for experiment in experiments:
        models = get_folders(dir = f"{results_dir}/{experiment}")
        for model in models:
            results = process_experiment(results_dir, experiment, model)
            results.to_csv(f"{results_dir}/{experiment}/{model}/aggregated_results.csv")
            create_latex_table(results, results_dir, experiment, model)