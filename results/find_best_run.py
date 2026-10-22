import os
from os.path import exists, join
from load_results import load_all_runs_data
import numpy as np
import glob

def find_best_acc(dataset, model, subject):

    runs_path = join(dataset, model, subject)
    # best_mean = 0
    best_mean = np.inf
    best_hp_config = ''

    for folder_name in os.listdir(runs_path):
        hp_config = folder_name
        run_data = load_all_runs_data(model, dataset, subject, hp_config)

        acc = []
        losses = []

        for n, run in enumerate(run_data):
            val_losses = run_data[run]['val_losses.npy']
            losses.append(np.min(val_losses))

        mean = np.mean(losses)

        if mean < best_mean:
            best_hp_config = hp_config
            best_mean = mean

    run_data = load_all_runs_data(model, dataset, subject, best_hp_config)
    best_test_accs = []
    for n, run in enumerate(run_data):
        test_accs = run_data[run]['test_accs.npy']
        best_test_accs.append(test_accs[-1])


    return np.mean(best_test_accs), np.std(best_test_accs), best_hp_config

import csv
def save_results_individual(dataset, subjects):
    csv_results_filename = f"analysis_individual/{dataset}_results.csv"

    # find available models
    models = os.listdir(dataset)
    models = ['matt']

    # Check if file exists and remove it
    if not exists("analysis_individual"):
        os.mkdir("analysis_individual")
    elif exists(csv_results_filename):
        os.remove(csv_results_filename)


    # Create CSV file and write results
    with open(csv_results_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        collected_data = {}
        for model in models:
            print(model)
            collected_data[model] = []

        # Write header
        writer.writerow(['Subject'] + models)
        # Write data
        for s in subjects[dataset]:
            aggs = []
            for model in models:
                if not exists(join(dataset, model, f'subject_{s}')):
                    print(f"Warning: No results found for {dataset}/{model}/subject {s}")
                    continue
                    print(model)
                mean, std, hp = find_best_acc(dataset, model, f'subject_{s}')
                collected_data[model].append(mean)
                aggs.append(f"{mean:.4f} $\pm$ {std:.4f}")

            # Write row
            writer.writerow([f'Subject {s}'] + aggs)
        summary_data = []
        for model in models:
            if len(collected_data[model]) > 0:
                summary_data.append(np.round(np.mean(collected_data[model]),4))
            else:
                summary_data.append(0)
        writer.writerow([f'Summary'] + summary_data)
    print(f'Saved results for {dataset} in {csv_results_filename}')

def save_hps_individual(dataset, subjects):
    csv_hps_filename = f"analysis_individual/{dataset}_hps.csv"

    # find available models
    models = os.listdir(dataset)
    models = ['matt']

    # Check if file exists and remove it
    if not exists("analysis_individual"):
        os.mkdir("analysis_individual")
    elif exists(csv_hps_filename):
        os.remove(csv_hps_filename)

    # Create CSV file and write results
    with open(csv_hps_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['Subject'] + models)
        # Write data
        for s in subjects[dataset]:
            aggs = []
            for model in models:
                if not exists(join(dataset, model, f'subject_{s}')):
                    print(f"Warning: No HPs found for {dataset}/{model}/subject {s}")
                    continue
                _, _, hp = find_best_acc(dataset, model, f'subject_{s}')
                aggs.append(hp)
            # Write row
            writer.writerow([f'Subject {s}'] + aggs)
    print(f'Saved Hps for {dataset} in {csv_hps_filename}')
               
def save_results_joint(dataset, models):
    csv_results_filename = f"analysis_joint/results.csv"

    # Check if file exists and remove it
    if os.path.exists(csv_results_filename):
        os.remove(csv_results_filename)

    # Create CSV file and write results
    with open(csv_results_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['Method', 'MI', 'SSVEP', 'ERN'])
        
        # Write data
        for m in models:
            nosub_lst = []
            emb_lst = []
            idemb_lst = []
            head_lst = []
            writer.writerow([m])
            for data in dataset:
                print(m)
                if m == 'matt':
                    if data == 'bci':
                        n = '_bci'
                        nosub_mean, nosub_std, nosub_hp = find_best_acc(data, m + n + 'NoSub', 'subject_all')
                        emb_mean, emb_std, emb_hp = find_best_acc(data, m + n + 'Emb', 'subject_all')
                        idemb_mean, idemb_std, idemb_hp = find_best_acc(data, m + n + 'IdEmb', 'subject_all')
                        head_mean, head_std, head_hp = find_best_acc(data, m + n + 'Head', 'subject_all')
                    elif data == 'mamem':
                        n = '_mamem'
                        nosub_mean, nosub_std, nosub_hp = find_best_acc(data, m + n + 'NoSub', 'subject_all')
                        emb_mean, emb_std, emb_hp = find_best_acc(data, m + n + 'Emb', 'subject_all')
                        idemb_mean, idemb_std, idemb_hp = find_best_acc(data, m + n + 'IdEmb', 'subject_all')
                        head_mean, head_std, head_hp = find_best_acc(data, m + n + 'Head', 'subject_all')
                    elif data == 'bcicha':
                        n = '_cha'
                        nosub_mean, nosub_std, nosub_hp = find_best_acc(data, m + n + 'NoSub', 'subject_all')
                        emb_mean, emb_std, emb_hp = find_best_acc(data, m + n + 'Emb', 'subject_all')
                        idemb_mean, idemb_std, idemb_hp = find_best_acc(data, m + n + 'IdEmb', 'subject_all')
                        head_mean, head_std, head_hp = find_best_acc(data, m + n + 'Head', 'subject_all')
                else:
                    nosub_mean, nosub_std, nosub_hp = find_best_acc(data, m + 'NoSub', 'subject_all')
                    emb_mean, emb_std, emb_hp = find_best_acc(data, m + 'Emb', 'subject_all')
                    idemb_mean, idemb_std, idemb_hp = find_best_acc(data, m + 'IdEmb', 'subject_all')
                    head_mean, head_std, head_hp = find_best_acc(data, m + 'Head', 'subject_all')


                # Aggregate mean and std with a string
                nosub_agg = f"{nosub_mean:.4f} $\pm$ {nosub_std:.4f}"
                emb_agg = f"{emb_mean:.4f} $\pm$ {emb_std:.4f}"
                idemb_agg = f"{idemb_mean:.4f} $\pm$ {idemb_std:.4f}"
                head_agg = f"{head_mean:.4f} $\pm$ {head_std:.4f}"

                nosub_lst.append(nosub_agg)
                emb_lst.append(emb_agg)
                idemb_lst.append(idemb_agg)
                head_lst.append(head_agg)
                
            # Write row
            writer.writerow([f'NoSub', nosub_lst[0], nosub_lst[1], nosub_lst[2]])
            writer.writerow([f'Concat', emb_lst[0], emb_lst[1], emb_lst[2]])
            writer.writerow([f'IdEmb', idemb_lst[0], idemb_lst[1], idemb_lst[2]])
            writer.writerow([f'Head', head_lst[0], head_lst[1], head_lst[2]])


    print(f'Saved results for {dataset} in {csv_results_filename}')

def save_hps_joint(dataset, models):
    csv_results_filename = f"analysis_joint/hps.csv"

    # Check if file exists and remove it
    if os.path.exists(csv_results_filename):
        os.remove(csv_results_filename)

    # Create CSV file and write results
    with open(csv_results_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['Method' , 'MI' , 'SSVEP', 'ERN'])
        
        # Write data
        for m in models:
            nosub_lst = []
            emb_lst = []
            idemb_lst = []
            head_lst = []
            writer.writerow([m])
            for data in dataset:
                if m == 'matt':
                    if data == 'bci':
                        n = '_bci'
                        nosub_mean, nosub_std, nosub_hp = find_best_acc(data, m + n + 'NoSub', 'subject_all')
                        emb_mean, emb_std, emb_hp = find_best_acc(data, m + n + 'Emb', 'subject_all')
                        idemb_mean, idemb_std, idemb_hp = find_best_acc(data, m + n + 'IdEmb', 'subject_all')
                        head_mean, head_std, head_hp = find_best_acc(data, m + n + 'Head', 'subject_all')
                    elif data == 'mamem':
                        n = '_mamem'
                        nosub_mean, nosub_std, nosub_hp = find_best_acc(data, m + n + 'NoSub', 'subject_all')
                        emb_mean, emb_std, emb_hp = find_best_acc(data, m + n + 'Emb', 'subject_all')
                        idemb_mean, idemb_std, idemb_hp = find_best_acc(data, m + n + 'IdEmb', 'subject_all')
                        head_mean, head_std, head_hp = find_best_acc(data, m + n + 'Head', 'subject_all')
                    elif data == 'bcicha':
                        n = '_cha'
                        nosub_mean, nosub_std, nosub_hp = find_best_acc(data, m + n + 'NoSub', 'subject_all')
                        emb_mean, emb_std, emb_hp = find_best_acc(data, m + n + 'Emb', 'subject_all')
                        idemb_mean, idemb_std, idemb_hp = find_best_acc(data, m + n + 'IdEmb', 'subject_all')
                        head_mean, head_std, head_hp = find_best_acc(data, m + n + 'Head', 'subject_all')
                else:
                    nosub_mean, nosub_std, nosub_hp = find_best_acc(data, m + 'NoSub', 'subject_all')
                    emb_mean, emb_std, emb_hp = find_best_acc(data, m + 'Emb', 'subject_all')
                    idemb_mean, idemb_std, idemb_hp = find_best_acc(data, m + 'IdEmb', 'subject_all')
                    head_mean, head_std, head_hp = find_best_acc(data, m + 'Head', 'subject_all')

                nosub_lst.append(nosub_hp)
                emb_lst.append(emb_hp)
                idemb_lst.append(idemb_hp)
                head_lst.append(head_hp)

            # Write row
            writer.writerow([f'NoSub', nosub_lst[0], nosub_lst[1], nosub_lst[2]])
            writer.writerow([f'Concat', emb_lst[0], emb_lst[1], emb_lst[2]])
            writer.writerow([f'IdEmb', idemb_lst[0], idemb_lst[1], idemb_lst[2]])
            writer.writerow([f'Head', head_lst[0], head_lst[1], head_lst[2]])


    print(f'Saved results for {dataset} in {csv_results_filename}')
    
subjects = {'bci': [1,2,3,4,5,6,7,8,9],
            'mamem': [1,2,3,4,5,6,7,8,9,10,11],
            'bcicha': [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26]}

for dataset in ['mamem']:
   save_results_individual(dataset, subjects)
for dataset in ['mamem']:
   save_hps_individual(dataset, subjects)


dataset = ['bci', 'mamem', 'bcicha']
models = ['resnet', 'inception', 'inception_4L', 'matt']
save_results_joint(dataset,models)
save_hps_joint(dataset,models)


