import torch
import torch.nn as nn
from utils.functions import trainNetwork, testNetwork, testNetwork_auc
from mAtt.inception import InceptionModel, InceptionModelNoSub, InceptionModelEmb, InceptionModelIdEmb, InceptionModelHead
from mAtt.inception import InceptionModel_4L, InceptionModelNoSub_4L, InceptionModelEmb_4L, InceptionModelIdEmb_4L, InceptionModelHead_4L
from mAtt.resnet import ResNet, ResNetNoSub, ResNetHead, ResNetEmbedding, ResNetIdEmbedding
from mAtt.rocket import run_rocket
from mAtt.mAtt import mAtt_bci, mAtt_mamem, mAtt_cha
from mAtt.mAtt import mAtt_bciNoSub, mAtt_bciHead, mAtt_bciEmb, mAtt_bciIdEmb
from mAtt.mAtt import mAtt_mamemNoSub, mAtt_mamemHead, mAtt_mamemEmb, mAtt_mamemIdEmb
from mAtt.mAtt import mAtt_chaNoSub, mAtt_chaHead, mAtt_chaEmb, mAtt_chaIdEmb
from mAtt.eegTCNet import EEGTCNet, EEGTCNetNoSub, EEGTCNetHead, EEGTCNetEmbedding, EEGTCNetIdEmbedding
from utils.bciDataLoader import BCIDataLoader
from utils.bciDataLoaderFull import BCIDataLoaderFull
from utils.mamemDataloader import MAMEMDataLoader, MAMEMDataLoader_fixed
from utils.mamemDataloaderFull import MAMEMDataLoaderFull, MAMEMDataLoaderFull_fixed
from utils.bcichaDataLoader import BCIchaDataLoader, BCIchaDataLoader_fixed
from utils.bcichaDataLoaderFull import BCIchaDataLoaderFull, BCIchaDataLoaderFull_fixed


import os
import argparse

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cpu', help='Torch Device for computations')
    ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
    ap.add_argument('--sub', type=str, default='all', help='subjectxx you want to train')
    ap.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-1, help='weight decay')
    ap.add_argument('--iterations', type=int, default=500, help='number of training iterations')
    ap.add_argument('--epochs', type=int, default=3, help='number of epochs that you want to use for split EEG signals')
    ap.add_argument('--bs', type=int, default=16, help='batch size')
    ap.add_argument('--model_path', type=str, default='./checkpoint/BCIcha/', help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default='./data/BCIcha/', help='data path')
    ap.add_argument('--model', type=str, default='matt_bciIdEmb', help='model name')
    ap.add_argument('--dataset', type=str, default='bci', help='dataset name')
    ap.add_argument('--subject_weights', type=float, default=0.01, help='subject weights')
    ap.add_argument('--rocket_kernel', type=int, default=10000, help='Number of kernels for ROCKET')
    args = vars(ap.parse_args())

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
    for repeat in range(1, args['repeat']+1):
        print(f'+++++++++++ Repeat: {repeat} +++++++++++ ')
        if args['dataset'] == 'bci':
            in_channels = 22
            num_pred_classes = 4
            num_subjects = 9
            args['data_path'] = './data/BCICIV_2a_mat/'
            if args['sub'] == 'all':
                dataloader = BCIDataLoaderFull(subject=args['sub'], ratio=8, data_path=args['data_path'], bs=args['bs'],
                                               model=args['model'], subject_weights=args['subject_weights'])
            else:
                dataloader = BCIDataLoader(subject=args['sub'], ratio=8, data_path=args['data_path'], bs=args['bs'], model=args['model'])
            trainloader, validloader, testloader = dataloader.get_dataloader()

        elif args['dataset'].startswith('mamem'):
            in_channels = 8
            num_pred_classes  = 5
            num_subjects = 11
            args['data_path'] = './data/MAMEM/'
            if args['dataset'].endswith('fixed'):
                if args['sub'] == 'all':
                    dataloader = MAMEMDataLoaderFull_fixed(subject=args['sub'], ratio=8, data_path=args['data_path'],
                                                     bs=args['bs'],
                                                     model=args['model'], subject_weights=args['subject_weights'])
                else:
                    dataloader = MAMEMDataLoader_fixed(subject=args['sub'], ratio=8, data_path=args['data_path'], bs=args['bs'],
                                                 model=args['model'])
                trainloader, validloader, testloader = dataloader.get_dataloader()
            else:
                if args['sub'] == 'all':
                    dataloader = MAMEMDataLoaderFull(subject=args['sub'], ratio=8, data_path=args['data_path'], bs=args['bs'],
                                                     model=args['model'], subject_weights=args['subject_weights'])
                else:
                    dataloader = MAMEMDataLoader(subject=args['sub'], ratio=8, data_path=args['data_path'], bs=args['bs'], model=args['model'])
                trainloader, validloader, testloader = dataloader.get_dataloader()

        elif args['dataset'].startswith('bcicha'):
            in_channels = 56
            num_pred_classes = 2
            num_subjects = 16
            args['data_path'] = './data/BCIcha/'
            if args['dataset'].endswith('fixed'):
                if args['sub'] == 'all':
                    dataloader = BCIchaDataLoaderFull_fixed(subject=args['sub'], data_path=args['data_path'], bs=args['bs'],
                                                      model=args['model'], subject_weights=args['subject_weights'])
                else:
                    dataloader = BCIchaDataLoader_fixed(subject=args['sub'], data_path=args['data_path'], bs=args['bs'], model=args['model'])
                trainloader, validloader, testloader = dataloader.get_dataloader()
            else:
                if args['sub'] == 'all':
                    dataloader = BCIchaDataLoaderFull(subject=args['sub'], data_path=args['data_path'], bs=args['bs'],
                                                      model=args['model'], subject_weights=args['subject_weights'])
                else:
                    dataloader = BCIchaDataLoader(subject=args['sub'], data_path=args['data_path'], bs=args['bs'], model=args['model'])
                trainloader, validloader, testloader = dataloader.get_dataloader()
        else:
            raise ValueError('No such dataset')

        print(f'subject {args["sub"]}')

        #device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if args['model'] == 'inception':
            net = InceptionModel(in_channels=in_channels, num_pred_classes=num_pred_classes)
        elif args['model'] == 'inceptionNoSub':
            net = InceptionModelNoSub(in_channels=in_channels, num_pred_classes=num_pred_classes)
        elif args['model'] == 'inceptionHead':
            net = InceptionModelHead(in_channels=in_channels, num_pred_classes=num_pred_classes, num_subjects=num_subjects)
        elif args['model'] == 'inceptionEmb':
            net = InceptionModelEmb(in_channels=in_channels, num_pred_classes=num_pred_classes, num_subjects=num_subjects)
        elif args['model'] == 'inceptionIdEmb':
            net = InceptionModelIdEmb(in_channels=in_channels, num_pred_classes=num_pred_classes, num_subjects=num_subjects)

        elif args['model'] == 'inception_4L':
            net = InceptionModel_4L(in_channels=in_channels, num_pred_classes=num_pred_classes)
        elif args['model'] == 'inception_4LNoSub':
            net = InceptionModelNoSub_4L(in_channels=in_channels, num_pred_classes=num_pred_classes)
        elif args['model'] == 'inception_4LHead':
            net = InceptionModelHead_4L(in_channels=in_channels, num_pred_classes=num_pred_classes, num_subjects=num_subjects)
        elif args['model'] == 'inception_4LEmb':
            net = InceptionModelEmb_4L(in_channels=in_channels, num_pred_classes=num_pred_classes, num_subjects=num_subjects)
        elif args['model'] == 'inception_4LIdEmb':
            net = InceptionModelIdEmb_4L(in_channels=in_channels, num_pred_classes=num_pred_classes, num_subjects=num_subjects)

        elif args['model'] == 'resnet':
            net = ResNet(in_channels=in_channels, num_pred_classes=num_pred_classes)
        elif args['model'] == 'resnetNoSub':
            net = ResNetNoSub(in_channels=in_channels, num_pred_classes=num_pred_classes)
        elif args['model'] == 'resnetHead':
            net = ResNetHead(in_channels=in_channels, num_pred_classes=num_pred_classes, num_subjects=num_subjects)#.to(device)
        elif args['model'] == 'resnetEmb':
            net = ResNetEmbedding(in_channels=in_channels, num_pred_classes=num_pred_classes, num_subjects=num_subjects)#.to(device)
        elif args['model'] == 'resnetIdEmb':
            net = ResNetIdEmbedding(in_channels=in_channels, num_pred_classes=num_pred_classes, num_subjects=num_subjects)

        elif args['model'] == 'TCNet':
            net = EEGTCNet(args['dataset'], Chans=in_channels, nb_classes=num_pred_classes)
        elif args['model'] == 'TCNetNoSub':
            net = EEGTCNetNoSub(args['dataset'], Chans=in_channels, nb_classes=num_pred_classes)
        elif args['model'] == 'TCNetHead':
            net = EEGTCNetHead(args['dataset'], Chans=in_channels, nb_classes=num_pred_classes, num_subjects=num_subjects)
        elif args['model'] == 'TCNetEmb':
            net = EEGTCNetEmbedding(args['dataset'], Chans=in_channels, nb_classes=num_pred_classes)
        elif args['model'] == 'TCNetIdEmb':
            net = EEGTCNetIdEmbedding(args['dataset'], Chans=in_channels, nb_classes=num_pred_classes, num_subjects=num_subjects)

        elif args['model'] == 'matt_mamemNoSub':
            net = mAtt_mamemNoSub(7)
        elif args['model'] == 'matt_mamemHead':
            net = mAtt_mamemHead(7)
        elif args['model'] == 'matt_mamemEmb':
            net = mAtt_mamemEmb(7)
        elif args['model'] == 'matt_mamemIdEmb':
            net = mAtt_mamemIdEmb(7)

        elif args['model'] == 'matt_bciNoSub':
            net = mAtt_bciNoSub(args['epochs'])
        elif args['model'] == 'matt_bciHead':
            net = mAtt_bciHead(args['epochs'])
        elif args['model'] == 'matt_bciEmb':
            net = mAtt_bciEmb(args['epochs'])
        elif args['model'] == 'matt_bciIdEmb':
            net = mAtt_bciIdEmb(args['epochs'])

        elif args['model'] == 'matt_chaNoSub':
            net = mAtt_chaNoSub(args['epochs'])
        elif args['model'] == 'matt_chaHead':
            net = mAtt_chaHead(args['epochs'])
        elif args['model'] == 'matt_chaEmb':
            net = mAtt_chaEmb(args['epochs'])
        elif args['model'] == 'matt_chaIdEmb':
            net = mAtt_chaIdEmb(args['epochs'])

        elif args['model'] == 'rocket':
            run_rocket(dataloader, args['rocket_kernel'], args['dataset'], tune=True)
        elif args['model'] == 'matt':
            if args['dataset'] == 'bci':
                net = mAtt_bci(args['epochs']).cpu()
            elif args['dataset'] == 'mamem':
                net = mAtt_mamem(7).cpu()
            elif args['dataset'] == 'bcicha':
                net = mAtt_cha(args['epochs']).cpu()
        else:
            raise ValueError('No such model')

        if args['model'] != 'rocket':
            net = net.to(args['device'])
            args.pop('data_path')

            trainNetwork(net,
                        trainloader,
                        validloader,
                        testloader,
                        **args
                        )


            net = torch.load(os.path.join(args["model_path"], f'repeat{args["repeat"]}_sub{args["sub"]}_epochs{args["epochs"]}_lr{args["lr"]}_wd{args["wd"]}.pt'))
            if (args['dataset'].startswith('mamem')) or (args['dataset'] == 'bci') or (args['dataset'] == 'bci_fixed'):
                acc = testNetwork(net, testloader, args["device"])
                print(f'Test acc: {acc * 100:.2f}')
            elif args['dataset'].startswith('bcicha'):
                auc = testNetwork_auc(net, testloader, args["device"])
                print(f'Test auc {auc * 100:.2f}')
            else:
                raise ValueError('No such dataset')




