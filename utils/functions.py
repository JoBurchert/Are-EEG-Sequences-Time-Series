import torch
import torch.nn as nn
import sys
import os
sys.path.append("..") 
from models.optimizer import MixOptimizer
from sklearn.metrics import roc_auc_score as ras
import numpy as np
from utils.save_results import save_output

def trainNetwork(net, trainloader, validloader, testloader, model_path=None, model=None, dataset=None, bs=64,
                 iterations=500, lr=5*1e-4, wd=None, repeat=None, sub=None, epochs=None, subject_weights=None,
                 verbose=True, save_model=True, save_results=True, early_stopping=True, grace_period=20, **kwargs):

    device = kwargs['device']
    softmax = nn.Softmax(dim=1)
    CE = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
    optimizer = MixOptimizer(optimizer)
    bestLoss = 1e10
    bestLoss_early_stopping = 1e10
    val_len = len(validloader)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_accs = []

    no_improvement_count = 0

    for ite in range(iterations):
        net.train()
        acc_val = 0
        acc_tr = 0
        tr_len = 0
        for row in trainloader:
            xb, yb = row[:-1], row[-1]
            tr_len += yb.shape[0]
            xb = [x.to(device) for x in xb]
            yb = yb.to(device)
            out = net(*xb)
            loss = CE(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_tr += (torch.max(out, 1).indices==yb).sum().item()
        
        net.eval()
        TL = 0
        for row in validloader:
            xb, yb = row[:-1], row[-1]
            xb = [x.to(device) for x in xb]
            yb = yb.to(device)
            with torch.no_grad():
                out = net(*xb)
                if torch.argmax(softmax(out)) == yb:
                    acc_val += 1

                TL += (CE(out, yb).item())
        if early_stopping:
            if TL < bestLoss_early_stopping:
                bestLoss_early_stopping = TL
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= grace_period:
                if verbose:
                    print(f'Early stopping at iteration {ite} as no improvement has been observed in {grace_period} '
                          f'iterations for the validation loss.')
                final_path = os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}.pt')
                torch.save(net, final_path)
                break

        if verbose:
            print('')
            print(f'Iteration{ite}=====')
            print(f'train_loss:{loss:.4f}    val_loss:{TL/val_len:.4f}')
            print(f'train_acc:{acc_tr/tr_len:.4f}    val_acc:{acc_val/val_len:.4f}')

        train_losses.append(loss.detach().cpu())
        train_accs.append(acc_tr/tr_len)
        val_losses.append(TL/val_len)
        val_accs.append(acc_val/val_len)

        if TL < bestLoss:
            if save_model:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                if verbose:
                    print(f'create {model_path}.......')
                bestLoss = TL
                final_path = os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}.pt')
                if verbose:
                    print(f'saving to {final_path}')
                torch.save(net, final_path)
                testnet = torch.load(final_path, weights_only=False)
                if dataset.startswith('bcicha'):
                    test_acc = testNetwork_auc(testnet, testloader, device)
                else:
                    test_acc = testNetwork(testnet, testloader, device)
            else:
                if dataset.startswith('bcicha'):
                    test_acc = testNetwork_auc(testnet, testloader, device)
                else:
                    test_acc = testNetwork(testnet, testloader, device)
            test_accs.append(test_acc)
            if verbose:
                print(f'test_acc:{test_acc}')
    if save_model:
        net = torch.load(os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}.pt'))
    net.train()

    if save_results:
        save_output(model_name=model, dataset=dataset, sub=sub, bs=bs, lr=lr, wd=wd, alpha=subject_weights,
                    results=[np.array(train_losses),
                             np.array(train_accs),
                             np.array(val_losses),
                             np.array(val_accs),
                             np.array(test_accs)],
                    **kwargs)

    return TL/val_len, acc_val/val_len # return last validation loss and acc
    # return net

def testNetwork(net, testloader, device):
    net.eval()
    acc = 0
    softmax = nn.Softmax(dim=1)
    for row in testloader:
        xb, yb = row[:-1], row[-1]
        xb = [x.to(device) for x in xb]
        yb = yb.to(device)
        with torch.no_grad():
            pred = net(*xb)
            if torch.argmax(softmax(pred)) == yb:
                acc += 1
    
    return acc / len(testloader)

def testNetwork_auc(net, testloader, device):
    net.eval()
    acc = 0
    softmax = nn.Softmax(dim=1)
    y_pred = torch.empty(0, device=device)
    y_true = torch.empty(0, device=device)
    for row in testloader:
        with torch.no_grad():
            xb, yb = row[:-1], row[-1]
            xb = [x.to(device) for x in xb]
            yb = yb.to(device)
            pred = net(*xb)
            y_pred = torch.cat((y_pred, pred[:, 1]), 0)
            y_true = torch.cat((y_true, yb), 0)

    return ras(y_true.detach().cpu().numpy(),
               y_pred.detach().cpu().numpy())



