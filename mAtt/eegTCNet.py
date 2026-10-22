# *----------------------------------------------------------------------------*
# * Copyright (C) 2020 ETH Zurich, Switzerland                                 *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Thorir Mar Ingolfsson                                             *
# *----------------------------------------------------------------------------*

import torch
import torch.nn as nn
from torch.nn.modules import Conv2d


class EEGTCNet(nn.Module):
    def __init__(self, dataset, nb_classes, Chans=64, layers=3, kernel_s=10, filt=10, dropout=0, activation='relu', F1=4, D=2,
                 kernLength=64, dropout_eeg=0.1):
        super().__init__()
        # input1 = Input(shape=(1, Chans, Samples))
        # input2 = Permute((3, 2, 1))(input1)
        regRate = .25
        numFilters = F1
        F2 = numFilters * D

        if dataset == "bcicha":
            kernel_height = 16
            pooling_window = 8
            linear_in_dim = 105
        elif dataset == "bci":
            kernel_height = 8
            pooling_window = 4
            linear_in_dim = 417
        elif dataset == "mamem":
            kernel_height = 6
            pooling_window = 3
            linear_in_dim = 118
        else:
            raise NotImplementedError
        self.linear_in_dim = linear_in_dim

        self.EEGNet_sep = EEGNet(input_size=1, F1=F1, kernLength=kernLength, kernel_height=kernel_height, pooling_window=pooling_window,
                                 D=D, Chans=Chans, dropout=dropout_eeg)

        self.outs = TCN_block(input_dimension=F2, depth=layers, kernel_size=kernel_s, filters=filt,
                              dropout=dropout, activation=activation)

        # TODO: Warning, this is currently hardcoded for BCIcha
        self.dense = nn.Linear(linear_in_dim, nb_classes)

    def forward(self, x:torch.Tensor):
        x = x.unsqueeze(1) # transform to an image
        x = self.EEGNet_sep(x)
        x = x[:, :, -1, :]

        x = self.outs(x)
        x = x[:, -1, :]

        x = self.dense(x)
        return x


class EEGTCNetIdEmbedding(nn.Module):
    def __init__(self, dataset, nb_classes, num_subjects, Chans=64, layers=3, kernel_s=10, filt=10, dropout=0, activation='relu', F1=4, D=2,
                 kernLength=64, dropout_eeg=0.1):
        super().__init__()
        regRate = .25
        numFilters = F1
        F2 = numFilters * D

        if dataset == "bcicha":
            kernel_height = 16
            pooling_window = 8
            linear_in_dim = 89
        elif dataset == "bci":
            kernel_height = 8
            pooling_window = 4
            linear_in_dim = 401
        elif dataset == "mamem":
            kernel_height = 8
            pooling_window = 4
            linear_in_dim = 102
        else:
            raise NotImplementedError

        emb_dim = 16
        self.subject_embeddings = nn.Embedding(num_subjects+1, emb_dim, padding_idx=0)

        self.EEGNet_sep = EEGNet(input_size=1, F1=F1, kernLength=kernLength, kernel_height=kernel_height, pooling_window=pooling_window,
                                 D=D, Chans=Chans+emb_dim, dropout=dropout_eeg)

        self.outs = TCN_block(input_dimension=F2, depth=layers, kernel_size=kernel_s, filters=filt,
                              dropout=dropout, activation=activation)

        # TODO: Warning, this is currently hardcoded for BCIcha
        self.dense = nn.Linear(linear_in_dim, nb_classes)

    def forward(self, x:torch.Tensor, x_sub:torch.Tensor):
        x_sub = x_sub.unsqueeze(1)
        z_sub = self.subject_embeddings(x_sub.int())
        z_sub = torch.squeeze(z_sub, dim=1).permute(0, 1, 3, 2)
        x = torch.cat([x.unsqueeze(1), z_sub], dim=2)
        x = self.EEGNet_sep(x)
        x = x[:, :, -1, :]

        x = self.outs(x)
        x = x[:, -1, :]

        x = self.dense(x)
        return x


class EEGTCNetEmbedding(nn.Module):
    def __init__(self, dataset, nb_classes, Chans=64, layers=3, kernel_s=10, filt=10, dropout=0, activation='relu', F1=4, D=2,
                 kernLength=64, dropout_eeg=0.1):
        super().__init__()
        numFilters = F1
        F2 = numFilters * D

        if dataset == "bcicha":
            kernel_height = 16
            pooling_window = 8
            linear_in_dim = 105
        elif dataset == "bci":
            kernel_height = 8
            pooling_window = 4
            linear_in_dim = 417
        elif dataset == "mamem":
            kernel_height = 8
            pooling_window = 4
            linear_in_dim = 118
        else:
            raise NotImplementedError

        self.EEGNet_sep = EEGNet(input_size=1, F1=F1, kernLength=kernLength, kernel_height=kernel_height, pooling_window=pooling_window,
                                 D=D, Chans=Chans, dropout=dropout_eeg)

        self.outs = TCN_block(input_dimension=F2, depth=layers, kernel_size=kernel_s, filters=filt,
                              dropout=dropout, activation=activation)

        # TODO: Warning, this is currently hardcoded for BCIcha
        self.dense = nn.Linear(linear_in_dim, nb_classes)

    def forward(self, x:torch.Tensor, x_sub:torch.Tensor):
        x = torch.cat([x, x_sub], dim=1)
        x = x.unsqueeze(1)
        x = self.EEGNet_sep(x)
        x = x[:, :, -1, :]

        x = self.outs(x)
        x = x[:, -1, :]

        x = self.dense(x)
        return x


class EEGTCNetNoSub(EEGTCNet):
    def forward(self, x:torch.Tensor, x_sub:torch.Tensor):
        # Ignoring the subject ID
        x = x.unsqueeze(1)
        x = self.EEGNet_sep(x)
        x = x[:, :, -1, :]

        x = self.outs(x)
        x = x[:, -1, :]

        x = self.dense(x)
        return x

class EEGTCNetHead(EEGTCNet):
    def __init__(self, dataset, nb_classes, num_subjects, Chans=64, layers=3, kernel_s=10, filt=10, dropout=0, activation='relu', F1=4, D=2,
                 kernLength=64, dropout_eeg=0.1):
        super().__init__(dataset, nb_classes, Chans, layers, kernel_s, filt, dropout, activation, F1, D, kernLength, dropout_eeg)
        sub_emb_size = 3
        self.sub_embedding = nn.Sequential(*[
            nn.Linear(num_subjects, 6),
            nn.Tanh(),
            nn.Linear(6, sub_emb_size),
        ])

        # recreate head
        self.linear_in_dim += sub_emb_size
        # TODO: Warning, this is currently hardcoded for BCIcha
        self.dense = nn.Linear(self.linear_in_dim, nb_classes)

    def forward(self, x: torch.Tensor, x_sub: torch.Tensor):
        # Ignoring the subject ID
        x = x.unsqueeze(1)
        x = self.EEGNet_sep(x)
        x = x[:, :, -1, :]

        x = self.outs(x)
        x = x[:, -1, :]

        sub_emb = self.sub_embedding(x_sub)
        x = torch.cat([x, sub_emb], dim=1)
        x = self.dense(x)
        return x

def EEGNet(input_size, F1=4, kernLength=64, kernel_height=16, pooling_window=8, D=2, Chans=22, dropout=0.1):
    F2 = F1 * D
    model = nn.Sequential(
        nn.Conv2d(input_size, F1, kernel_size=(kernLength, 1), padding="same", bias=False),
        nn.BatchNorm2d(F1),
        nn.Conv2d(F1, F2, kernel_size=(1, Chans), groups=F1, bias=False), # Depthwise Conv2D
        nn.BatchNorm2d(F2),
        nn.ELU(),
        nn.AvgPool2d(kernel_size=(pooling_window, 1)),
        nn.Dropout(p=dropout),
        Conv2d(F2, F2, kernel_size=(kernel_height, 1), stride=1, padding=(kernel_height//2, 0)), # Separable Conv2D
        nn.BatchNorm2d(F2),
        nn.ELU(),
        nn.AvgPool2d(kernel_size=(pooling_window, 1)),
        nn.Dropout(p=dropout),
    )
    return model


class TCN_block(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout, activation='relu'):
        super().__init__()
        self.input_dimension = input_dimension
        self.filters = filters

        self.first_layer = nn.Sequential(
            nn.Conv1d(input_dimension, filters, kernel_size=kernel_size, padding="same"),  # to keep dims compatible in the residual connection
            # nn.Conv1d(input_dimension, filters, kernel_size=kernel_size, padding=kernel_size // 2),  # trying to copy "causal" padding from TF
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(filters, filters, kernel_size=kernel_size, padding="same"),  # to keep dims compatible in the residual connection
            # nn.Conv1d(filters, filters, kernel_size=kernel_size, padding=kernel_size // 2),  # trying to copy "causal" padding from TF
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if input_dimension != filters:
            self.first_layer_postfix = nn.Conv1d(input_dimension, filters, kernel_size=1, padding="same")

        self.other_layers = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.other_layers.append(nn.Sequential(
                nn.Conv1d(filters, filters, kernel_size=kernel_size, dilation=2 ** (i + 1), padding="same"),
                # nn.Conv1d(filters, filters, kernel_size=kernel_size, dilation=2 ** (i + 1), padding=kernel_size // 2),  # trying to copy "causal" padding from TF
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(filters, filters, kernel_size=kernel_size, dilation=2 ** (i + 1), padding="same"),
                # nn.Conv1d(filters, filters, kernel_size=kernel_size, dilation=2 ** (i + 1), padding=kernel_size // 2),  # trying to copy "causal" padding from TF
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))


    def forward(self, x):
        z = self.first_layer(x)
        if self.input_dimension != self.filters:
            x = self.first_layer_postfix(x)
        z = torch.nn.functional.relu(z + x)

        for layer in self.other_layers:
            z_layer = layer(z)
            z = torch.nn.functional.relu(z_layer + z)
        return z

