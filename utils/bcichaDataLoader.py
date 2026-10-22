import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from scipy import io
import os


class BCIchaDataLoader:
    def __init__(self, subject, data_path='./data/BCIcha/', bs=64, model=None, dev='cpu'):
        self.subject = subject
        self.data_path = data_path
        self.bs = bs
        self.model = model
        self.dev = torch.device(dev)
        self.timeseries_length = 160
        # Uncomment the following line if you want to use GPU
        # self.dev = torch.device("cuda")

    def get_dataloader(self):
        train = io.loadmat(os.path.join(self.data_path, f'Data_S{int(self.subject):02d}_Sess' + '.mat'))

        if self.model == 'matt':
            tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
        else:
            tempdata = torch.Tensor(train['x_test'])
        templabel = torch.Tensor(train['y_test']).view(-1)
        x_train = tempdata[:180]
        y_train = templabel[:180]

        x_valid = tempdata[180:240]
        y_valid = templabel[180:240]

        x_test = tempdata[240:340]
        y_test = templabel[240:340]

        self.x_train = x_train.to(self.dev)
        self.y_train = y_train.long().to(self.dev)
        self.x_valid = x_valid.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.x_test = x_test.to(self.dev)
        self.y_test = y_test.long().to(self.dev)

        train_dataset = Data.TensorDataset(self.x_train, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.y_test)

        trainloader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        validloader = Data.DataLoader(
            dataset=valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        testloader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        return trainloader, validloader, testloader

from sklearn.model_selection import train_test_split

class BCIchaDataLoader_fixed:
    def __init__(self, subject, data_path='./data/BCIcha/', bs=64, model=None, dev='cpu'):
        self.subject = subject
        self.data_path = data_path
        self.bs = bs
        self.model = model
        self.dev = torch.device(dev)
        self.timeseries_length = 160

    def get_dataloader(self):
        train = io.loadmat(os.path.join(self.data_path, f'Data_S{int(self.subject):02d}_Sess' + '.mat'))

        if self.model == 'matt':
            tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
        else:
            tempdata = torch.Tensor(train['x_test'])
        templabel = torch.Tensor(train['y_test']).view(-1)

        # Split the data into train, validation, and test sets with random splits while maintaining the same sizes
        x_train, x_test, y_train, y_test = train_test_split(tempdata, templabel, test_size=0.47, )
        x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.625, )

        self.x_train = x_train.to(self.dev)
        self.y_train = y_train.long().to(self.dev)
        self.x_valid = x_valid.to(self.dev)
        self.y_valid = y_valid.long().to(self.dev)
        self.x_test = x_test.to(self.dev)
        self.y_test = y_test.long().to(self.dev)

        train_dataset = Data.TensorDataset(self.x_train, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.y_test)

        trainloader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        validloader = Data.DataLoader(
            dataset=valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        testloader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        return trainloader, validloader, testloader
