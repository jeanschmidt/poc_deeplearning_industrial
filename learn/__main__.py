# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import sys
import warnings

import shutil

from setproctitle import setproctitle

import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from train import train
from test import test_print_save

from dataset.loader import JMDSAmendoasLoader


def main():
    warnings.filterwarnings('ignore')
    setproctitle('jmds.Almonds')
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))

    model = Net()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=model.learning_rate,
        momentum=model.momentum
    )

    if torch.cuda.is_available():
        print(" * CUDA AVAILABLE, USING IT")
        model = model.cuda()
        criterion = criterion.cuda()

    print(" * LOADING DATASET")

    jal = JMDSAmendoasLoader()
    train_batches = jal.get_train_batches(40)

    print(" * RUNNING TEACHING PROCESS")

    for epoch in range(1, 30):
        train(
            model,
            optimizer,
            criterion,
            epoch,
            train_batches
        )
        test_print_save(
            model,
            criterion,
            epoch,
            train_batches,
            jal
        )


if __name__ == '__main__':
    main()
