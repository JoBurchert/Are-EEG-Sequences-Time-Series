import pandas as pd
import torch
import torch.nn as nn
from utils.functions import trainNetwork, testNetwork, testNetwork_auc
from mAtt.inception import InceptionModel, InceptionModelEmb, InceptionModelIdEmb
from mAtt.resnet import ResNet, ResNetHead, ResNetEmbedding, ResNetIdEmbedding
from mAtt.rocket import run_rocket
from mAtt.mAtt import mAtt_bci, mAtt_mamem, mAtt_cha
from mAtt.vision_transformer import get_vit
from mAtt.ts_mixer import get_ts_mixer
from utils.bciDataLoader import BCIDataLoader
from utils.bciDataLoaderFull import BCIDataLoaderFull
from utils.mamemDataloader import MAMEMDataLoader
from utils.mamemDataloaderFull import MAMEMDataLoaderFull
from utils.bcichaDataLoader import BCIchaDataLoader
from utils.bcichaDataLoaderFull import BCIchaDataLoaderFull
from tqdm import tqdm

import os
import argparse

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cpu', help='Torch Device for computations')
    ap.add_argument('--iterations', type=int, default=50, help='number of training iterations')
    ap.add_argument('--epochs', type=int, default=3, help='number of epochs that you want to use for split EEG signals')
    ap.add_argument('--model_path', type=str, default='./checkpoint/BCIcha/', help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default='./data/BCIcha/', help='data path')
    ap.add_argument('--model', type=str, default='vit', help='model name')
    ap.add_argument('--dataset', type=str, default='bcicha', help='dataset name')
    ap.add_argument('--subject_weights', type=float, default=0.01, help='subject weights')
    ap.add_argument('--rocket_kernel', type=int, default=10000, help='Number of kernels for ROCKET')
    args = vars(ap.parse_args())
    args["model"] = "vit"

    print(f"Using Device: {args['device']}")

    """
    Datasets:
    BCI
        Input: 22  Cls: 4
    Mamem
        Input: 8 Cls: 5
    BCICHA 
        Input: 56  Cls: 2
    """
    subject_ids = {
        "bci": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "bcicha": [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26],
        "mamem": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }

    # data = {"lr": [], "wd": [],  "bs": [],  "depth": [], "mlp_dim": [], "dropout": [],
    #         "val_loss": [], "val_acc": []}

    # runs: 96 * 3
    for sub_id in subject_ids[args["dataset"]]:
        args["sub"] = sub_id
        for depth in tqdm([2, 4]):
            for mlp_dim in [256, 512]:
                for dropout in [0.0, 0.1]:
                    for lr in [5e-6, 1e-5, 5e-4]:
                        for wd in [1e-5, 1e-4, 1e-3, 1e-2]:
                            for bs in [16]:
                                # mean_val_loss, mean_val_acc = 0.0, 0.0
                                for restart in range(3):

                                    if args['dataset'] == 'bci':
                                        in_channels = 22
                                        num_pred_classes = 4
                                        num_subjects = 9
                                        args['data_path'] = './data/BCICIV_2a_mat/'
                                        if args['sub'] == 'all':
                                            dataloader = BCIDataLoaderFull(subject=args['sub'], ratio=8, data_path=args['data_path'], bs=bs,
                                                                           model=args['model'], subject_weights=args['subject_weights'])
                                        else:
                                            dataloader = BCIDataLoader(subject=args['sub'], ratio=8, data_path=args['data_path'], bs=bs, model=args['model'])
                                        trainloader, validloader, testloader = dataloader.get_dataloader()

                                    elif args['dataset'] == 'mamem':
                                        in_channels = 8
                                        num_pred_classes  = 5
                                        num_subjects = 11
                                        args['data_path'] = './data/MAMEM/'
                                        if args['sub'] == 'all':
                                            dataloader = MAMEMDataLoaderFull(subject=args['sub'], ratio=8, data_path=args['data_path'], bs=bs,
                                                                             model=args['model'], subject_weights=args['subject_weights'])
                                        else:
                                            dataloader = MAMEMDataLoader(subject=args['sub'], ratio=8, data_path=args['data_path'], bs=bs, model=args['model'])
                                        trainloader, validloader, testloader = dataloader.get_dataloader()

                                    elif args['dataset'] == 'bcicha':
                                        in_channels = 56
                                        num_pred_classes = 2
                                        num_subjects = 16
                                        args['data_path'] = './data/BCIcha/'
                                        if args['sub'] == 'all':
                                            dataloader = BCIchaDataLoaderFull(subject=args['sub'], data_path=args['data_path'], bs=bs,
                                                                              model=args['model'], subject_weights=args['subject_weights'])
                                        else:
                                            dataloader = BCIchaDataLoader(subject=args['sub'], data_path=args['data_path'], bs=bs, model=args['model'])
                                        trainloader, validloader, testloader = dataloader.get_dataloader()
                                    else:
                                        raise ValueError('No such dataset')

                                    net = get_vit(args['dataset'], in_channels=in_channels, num_pred_classes=num_pred_classes,
                                                  depth=depth, mlp_dim=mlp_dim, dropout=dropout)
                                    net = net.to(torch.device(args["device"]))

                                    args.pop('data_path')
                                    args['lr'] = lr
                                    args['wd'] = wd
                                    args['dropout'] = dropout
                                    args['bs'] = bs
                                    args['depth'] = depth
                                    args['mlp_dim'] = mlp_dim
                                    val_loss, val_acc = trainNetwork(
                                                net,
                                                trainloader,
                                                validloader,
                                                testloader,
                                                verbose=False,
                                                save_model=False,
                                                **args
                                                )
                                #     mean_val_loss += val_loss
                                #     mean_val_acc += val_acc
                                # mean_val_loss /= len(subject_ids[args["dataset"]])
                                # mean_val_acc /= len(subject_ids[args["dataset"]])
                                # data['lr'].append(lr)
                                # data['wd'].append(wd)
                                # data['dropout'].append(dropout)
                                # data['bs'].append(bs)
                                # data['depth'].append(depth)
                                # data['mlp_dim'].append(mlp_dim)
                                # data['val_loss'].append(mean_val_loss)
                                # data['val_acc'].append(mean_val_acc)
                                print(f"lr:{lr}, wd:{wd}, bs:{bs}, depth:{depth}, mlp_dim:{mlp_dim}, dropout:{dropout}")

                                # store intermediate results
                                # csv_file = f"logs/gs_{args['dataset']}_{args['model']}.csv"
                                # if os.path.exists(csv_file):
                                #     os.remove(csv_file)
                                # pd.DataFrame(data).to_csv(csv_file)
