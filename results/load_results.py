import os
import numpy as np


def load_all_runs_data(model_name, dataset, subject, hp_config):
    model_folder = os.path.join(dataset, model_name, subject, hp_config)

    # Find all run folders
    run_folders = []

    for folder_name in os.listdir(model_folder):
        if folder_name.startswith('run_'):
            run_folders.append(os.path.join(model_folder, folder_name))

    if not run_folders:
        print("No run folders found.")
        return None

    # Load data from all .npy files in all run folders
    all_loaded_data = {}
    for run_folder in run_folders:
        run_id = int(run_folder.split('_')[-1])
        loaded_data = {}

        for file_name in os.listdir(run_folder):
            if file_name.endswith('.npy'):
                file_path = os.path.join(run_folder, file_name)
                data = np.load(file_path)
                loaded_data[file_name] = data

        all_loaded_data[f'run_{run_id}'] = loaded_data
    return all_loaded_data

