# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable


def train(model, optimizer, criterion, epoch, train_batches):
    model.train()
    for batch_idx, get_data, get_target in train_batches:

        data = get_data()
        target = get_target()

        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            l2_reg = l2_reg.cuda()

        optimizer.zero_grad()
        output = model(data)

        l2_reg_copy = l2_reg.clone()
        for param in model.parameters():
            l2_reg_copy += (param * param).sum()
        loss = criterion(output, target) + \
            (l2_reg_copy * (model.l2_reg_increment * epoch))

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                'Train Epoch: {} [{} {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(train_batches),
                    len(train_batches),
                    100. * batch_idx / len(train_batches),
                    loss.data[0]
                )
            )
