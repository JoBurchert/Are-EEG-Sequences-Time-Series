import torch
import numpy as np

def symmetric(A):
    size = list(range(len(A.shape)))
    temp = size[-1]
    size.pop()
    size.insert(-1, temp)
    return 0.5 * (A + A.permute(*size))

def is_nan_or_inf(A):
    C1 = torch.nonzero(A == float('inf'))
    C2 = torch.nonzero(A != A)
    if len(C1.size()) > 0 or len(C2.size()) > 0:
        return True
    return False

def is_pos_def(x):
    return torch.all(torch.linalg.eigvals(x) > 0)

def matrix_operator(A, operator):
    u, s, v = A.svd()
    if operator == 'sqrtm':
        s.sqrt_()
    elif operator == 'rsqrtm':
        s.rsqrt_()
    elif operator == 'logm':
        s.log_()
    elif operator == 'expm':
        s.exp_()
    else:
        raise('operator %s is not implemented' % operator)
    
    output = u.mm(s.diag().mm(u.t()))
    
    return output

def tangent_space(A, ref, inverse_transform=False):
    ref_sqrt = matrix_operator(ref, 'sqrtm')
    ref_sqrt_inv = matrix_operator(ref, 'rsqrtm')
    middle = ref_sqrt_inv.mm(A.mm(ref_sqrt_inv))
    if inverse_transform:
        middle = matrix_operator(middle, 'logm')
    else:
        middle = matrix_operator(middle, 'expm')
    out = ref_sqrt.mm(middle.mm(ref_sqrt))
    return out

def untangent_space(A, ref):
    return tangent_space(A, ref, True)

def parallel_transform(A, ref1, ref2):
    print(A.size(), ref1.size(), ref2.size())
    out = untangent_space(A, ref1)
    out = tangent_space(out, ref2)
    return out

def orthogonal_projection(A, B):
    out = A - B.mm(symmetric(B.transpose(0,1).mm(A)))
    return out

def retraction(A, ref):
    data = A + ref
    Q, R = data.qr()
    # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out


import torch
from torch import nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)