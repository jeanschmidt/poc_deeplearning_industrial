# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    @property
    def drop_probability(self):
        return 0.5

    @property
    def learning_rate(self):
        return 0.005

    @property
    def momentum(self):
        return 0.4

    @property
    def l2_reg_increment(self):
        return 0.01

    @property
    def input_dim(self):
        return 3

    @property
    def out_dim(self):
        return 2

    @property
    def conv1_dim(self):
        return 20

    @property
    def conv1_width(self):
        return 5

    @property
    def conv1_max_pool(self):
        return 7

    @property
    def conv2_dim(self):
        return 10

    @property
    def conv2_width(self):
        return 7

    @property
    def conv2_max_pool(self):
        return 7

    @property
    def hidden1_number(self):
        return 330

    @property
    def hidden2_number(self):
        return 50

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(self.input_dim, self.conv1_dim,
                               kernel_size=self.conv1_width)
        self.drop1 = nn.Dropout(self.drop_probability)
        self.pool1 = nn.MaxPool2d(self.conv1_max_pool)

        self.conv2 = nn.Conv2d(self.conv1_dim, self.conv2_dim,
                               kernel_size=self.conv2_width)
        self.drop2 = nn.Dropout(self.drop_probability)
        self.pool2 = nn.MaxPool2d(self.conv2_max_pool)

        self.fc1 = nn.Linear(self.hidden1_number, self.hidden2_number)
        self.fc2 = nn.Linear(self.hidden2_number, self.out_dim)

    def forward(self, x):
        x = F.relu(
            self.pool1(
                self.drop1(self.conv1(x))
            )
        )
        x = F.relu(
            self.pool2(
                self.drop2(self.conv2(x))
            )
        )
        x = x.view(-1, self.hidden1_number)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)  # sigmoid on last layer applyed by BCEWithLogitsLoss
        return x
