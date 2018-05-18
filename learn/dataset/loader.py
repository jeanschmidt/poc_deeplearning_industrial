# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import random
import torch

from skimage import io


def _randomized_file_list(dir_path):
    dir_list = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]
    random.shuffle(dir_list)
    return dir_list


class JMDSAmendoasLoader(object):
    def __init__(self):
        pass

    @property
    def percentage_test(self):
        return .93

    @property
    def data_good_path(self):
        return 'data/contrast/good'

    @property
    def data_bad_path(self):
        return 'data/contrast/bad'

    @property
    def random_good_list(self):
        if not hasattr(self, '_random_good_list'):
            self._random_good_list = _randomized_file_list(
                self.data_good_path
            )
        return self._random_good_list

    @property
    def random_bad_list(self):
        if not hasattr(self, '_random_bad_list'):
            self._random_bad_list = _randomized_file_list(
                self.data_bad_path
            )
        return self._random_bad_list

    @property
    def img_good_list(self):
        if not hasattr(self, '_img_good_list'):
            self._img_good_list = [
                torch.from_numpy(io.imread(x).T) for x in self.random_good_list
            ]
        return self._img_good_list

    @property
    def img_bad_list(self):
        if not hasattr(self, '_img_bad_list'):
            self._img_bad_list = [
                torch.from_numpy(io.imread(x).T) for x in self.random_bad_list
            ]
        return self._img_bad_list

    @property
    def img_dataset(self):
        if not hasattr(self, '_img_dataset'):
            self._img_dataset = [
                [torch.Tensor([1.0, 0.0, ]).float(), x, ]
                for x in self.img_good_list
            ] + [
                [torch.Tensor([0.0, 1.0, ]).float(), x, ]
                for x in self.img_bad_list
            ]
            random.shuffle(self._img_dataset)
        return self._img_dataset

    @property
    def train_dataset(self):
        if not hasattr(self, '_train_dataset'):
            self._train_dataset = self.img_dataset[
                :int(len(self.img_dataset) * self.percentage_test)
            ]
        return self._train_dataset

    @property
    def test_dataset(self):
        if not hasattr(self, '_test_dataset'):
            self._test_dataset = self.img_dataset[len(self.train_dataset):]
            self._test_dataset = (
                torch.stack([d[1] for d in self._test_dataset]).float(),
                torch.stack([d[0] for d in self._test_dataset]).float(),
            )
        return self._test_dataset

    def get_train_batches(self, batch_size):
        return self._build_batches_for(batch_size, self.train_dataset)

    def get_test_batches(self, batch_size):
        return self._build_batches_for(batch_size, self.test_dataset)

    def _build_batches_for(self, batch_size, dataset):
        def mk_get_data(bid):
            def get_data():
                return torch.stack([
                    d[1]
                    for d in dataset[(bid * batch_size):((bid + 1) * batch_size)]
                ]).float()
            return get_data
        def mk_get_target(bid):
            def get_target():
                return torch.stack([
                    d[0]
                    for d in dataset[(bid * batch_size):((bid + 1) * batch_size)]
                ]).float()
            return get_target
        batches = [
            [bid, mk_get_data(bid),mk_get_target(bid),]
            for bid in xrange(int(len(dataset) / batch_size))
        ]
        return batches
