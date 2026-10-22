import os
import numpy as np

def save_output(model_name, dataset, sub, bs, lr, wd, alpha, results, **kwargs):
    base_folder = './results'
    subject = f'subject_{sub}'
    if sub == 'all':
        hp_config = f'bs{bs}_lr{lr}_wd{wd}_alpha{alpha}'
    else:
        hp_config = f'bs{bs}_lr{lr}_wd{wd}'
    if model_name == "mixer":
        hp_config = hp_config + f"_blocks{kwargs['blocks']}_ffdim{kwargs['ff_dim']}_dropout{kwargs['dropout']}"
    elif model_name == "vit":
        hp_config = hp_config + f"_depth{kwargs['depth']}_mlpdim{kwargs['mlp_dim']}_dropout{kwargs['dropout']}"
    model_folder = os.path.join(base_folder, dataset, model_name, subject, hp_config)

    run_id = 1
    run_folder = os.path.join(model_folder, f'run_{run_id}')

    while os.path.exists(run_folder):
        run_id += 1
        run_folder = os.path.join(model_folder, f'run_{run_id}')
    os.makedirs(run_folder)

    outputs = ['train_losses', 'train_accs', 'val_losses', 'val_accs', 'test_accs']

    for i, result in enumerate(results):
        # Save the output to a file in the run folder
        output_file = os.path.join(run_folder, f'{outputs[i]}.npy')
        np.save(output_file, result)

