from sktime.classification.kernel_based import RocketClassifier
from sktime.datasets import load_unit_test
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import os
import csv

def run_rocket(dataloader, kernel, dataset, tune):

    X_train = dataloader.x_train.numpy()
    y_train = dataloader.y_train.numpy()
    if tune:
        X_test = dataloader.x_valid.numpy()
        y_test = dataloader.y_valid.numpy()
    else:
        X_test = dataloader.x_test.numpy()
        y_test = dataloader.y_test.numpy()


    trf = RocketClassifier(num_kernels=kernel)
    trf.fit(X_train, y_train)
    print('Num Kernels:', kernel)
    y_pred = trf.predict(X_test)

    if dataset.startswith('bcicha'):
        score = roc_auc_score(y_test, y_pred)
    else:
        score = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy:{score:.4f}')

