# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import time

import torch
from torch.autograd import Variable


def chunks(l):
    n = 40
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_test_result(set_type, test_loss, correct, test_dataset_len,
                      evaluate_time):
    print(
        '\n"{}" set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%) '
        'Evaluate Time: {:.4f}s ({:.2f}ms per test)\n'.format(
            set_type,
            test_loss,
            correct,
            test_dataset_len,
            100. * correct / test_dataset_len,
            evaluate_time,
            (evaluate_time / test_dataset_len) * 1000
        )
    )


def test_data(model, criterion, data, target):
    start_time = time.time()

    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    model.eval()
    test_dataset_len = data.size()[0]
    output = model(data)
    test_loss = criterion(output, target).data.sum()
    pred = output.data.max(1, keepdim=True)[1]
    target_pred = target.max(1, keepdim=True)[1]
    correct = pred.eq(target_pred.data).sum()
    test_loss /= test_dataset_len

    evaluate_time = time.time() - start_time

    return {
        'test_loss': test_loss,
        'correct': correct,
        'test_dataset_len': test_dataset_len,
        'evaluate_time': evaluate_time,
    }


def _sum_dicts(total, inc):
    for k in inc.keys():
        total[k] = total.get(k, 0.0) + inc.get(k, 0.0)
    return total


def test(test_dataset, model, criterion):
    data_lst = list(chunks(test_dataset[0]))
    target_lst = list(chunks(test_dataset[1]))

    total = {}

    for idx in xrange(len(data_lst)):
        ret = test_data(
            model,
            criterion,
            Variable(data_lst[idx], volatile=True),
            Variable(target_lst[idx])
        )
        _sum_dicts(total, ret)

    return total


def test_train(train_dataset, model, criterion):
    total = {}

    for _, get_data, get_target in train_dataset:
        ret = test_data(
            model,
            criterion,
            Variable(get_data(), volatile=True),
            Variable(get_target())
        )
        _sum_dicts(total, ret)

    return total


def test_print_save(model, criterion, epoch, train_batches, jal):
    test_result = test(jal.test_dataset, model, criterion)
    print_test_result('TEST', **test_result)
    train_result = test_train(train_batches, model, criterion)
    print_test_result('TRAIN', **train_result)
