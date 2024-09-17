import pandas as pd
import os

def get_folders(dir: str = "") -> list:
    folders = os.listdir(f"results/{dir}/")
    return folders

def get_metrics():
    metrics = ['MSETrain', 'EnergyScoreTrain', 'CRPSTrain', 'Gaussian NLLTrain',
       'CoverageTrain', 'IntervalWidthTrain', 'MSEValidation',
       'EnergyScoreValidation', 'CRPSValidation', 'Gaussian NLLValidation',
       'CoverageValidation', 'IntervalWidthValidation', 'MSETest',
       'EnergyScoreTest', 'CRPSTest', 'Gaussian NLLTest', 'CoverageTest',
       'IntervalWidthTest']
    return metrics

def extract_name(folder_name):
    parts = folder_name.split('_')
    if len(parts) > 5:
        return parts[-2] + '_' + parts[-1]  # Extract the last two parts (e.g., 'sr_reparam')
    else:
        return parts[-1] 

def get_methods(results_dir:str, experiment:str, model:str) -> list:
    base_path = f"results/{results_dir}/{experiment}/{model}"
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    folders.sort()
    methods = [extract_name(f) for f in folders]
    return methods

def process_experiment(experiment: str, model: str) -> pd.DataFrame:
    metrics = get_metrics()
    methods = get_methods(results_dir, experiment, model)
    columns = pd.MultiIndex.from_product([methods, ["Mean", "Std"]], names = ["Method", "Statistic"])
    results_df = pd.DataFrame(index = metrics, columns = columns)
    base_path = f"results/{results_dir}/{experiment}/{model}"
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    subfolders.sort()
    for i,f in enumerate(subfolders):
        results = pd.read_csv(os.path.join(f"results/{results_dir}/{experiment}/{model}/{f}", "test.csv"), index_col=0)
        results = results.loc[metrics].astype("float32")
        mean = results.mean(axis = 1).round(2)
        std = results.std(axis = 1).round(3)
        # Assign the mean and std to the correct MultiIndex in the DataFrame

        results_df[(methods[i], "Mean")] = mean
        results_df[(methods[i], "Std")] = std
    return results_df

def create_latex_table(results_df: pd.DataFrame, experiment: str, model: str) -> str:
    # Initialize an empty DataFrame to store the formatted values
    formatted_df = pd.DataFrame()
    metrics = results_df.index[-6:] # Extract only test metrics
    methods = results_df.columns.levels[0]  # Methods are the top level of the columns MultiIndex

    # Create a new DataFrame with the method as the index and metrics as columns
    for metric in metrics:
        formatted_df[metric] = [
            f"{results_df.loc[metric, (method, 'Mean')]:.3f}"
            for method in methods
        ]
    formatted_df.index = methods
    latex_table = formatted_df.to_latex(escape=False)
    #  save to a file
    with open(f"results/{results_dir}/{experiment}/{model}/aggregated_results.tex", "w") as f:
        f.write(latex_table)




if __name__ == "__main__":
    results_dir = "optimal_hp_multiple_seeds"
    experiments = get_folders(dir = results_dir)
    for experiment in experiments:
        models = get_folders(dir = f"{results_dir}/{experiment}")
        for model in models:
            results = process_experiment(experiment, model)
            results.to_csv(f"results/{results_dir}/{experiment}/{model}/aggregated_results.csv")
            create_latex_table(results, experiment, model)