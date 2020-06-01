from __future__ import absolute_import, division, print_function

import os
from os.path import join as pjoin

from collections import OrderedDict as odict

import numpy as np
import torch

import nibabel as nib
from torch.utils.data import Dataset

from transforms import PartialNineCrop
from utils import copy_mmap, list_merge


class DefaultLoader(object):
    expansion = 1

    def __init__(self, root):
        self.root = root

    def __call__(self, sid):
        sample = nib.load(pjoin(self.root, sid+'.nii'))
        return copy_mmap(sample.get_data())


class ADNIDataset(Dataset):
    def __init__(self, labels, root, sid_dict, target_dict,
                 loader=None, transform=None):
        self.labels = labels

        self.sid_dict = sid_dict
        self._sid = list_merge(sid_dict.values())
        self.target_dict = target_dict

        self.loader = loader if loader is not None else DefaultLoader(root)
        self.transform = transform

    def __len__(self):
        return len(self._sid)

    def __getitem__(self, idx):
        sid = self._sid[idx]
        image = self.loader(sid)
        target = self.labels.index(self.target_dict[sid])

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def fold_split(k, running_k, labels, subject_indices, target_dict, seed=1234):
    labels = tuple(labels)
    assert labels in [('AD', 'NL'), ('AD', 'MCI', 'NL')]  # TODO: more tasks

    label_by = odict()
    for label in labels:
        label_by[label] = []
    for sid in subject_indices:
        d = target_dict[sid]
        if d not in labels:
            continue
        label_by[d] += [sid]

    np.random.seed(seed)
    for label in labels:
        np.random.shuffle(label_by[label])
    np.random.seed()

    train = odict()
    valid = odict()
    for label in labels:
        train[label] = []
        valid[label] = []

        spliter = np.linspace(0, len(label_by[label]), k+1, dtype=int)
        for j, sid in enumerate(label_by[label]):
            if spliter[running_k] <= j < spliter[running_k+1]:
                valid[label] += [sid]
            else:
                train[label] += [sid]
        valid[label] = sorted(valid[label])
        train[label] = sorted(train[label])

    def merge_dict(xdict):
        return [item for value in xdict.values() for item in value]

    print('Train', *zip(labels, map(len, train.values())),
          'Total', len(merge_dict(train)))
    print('Valid', *zip(labels, map(len, valid.values())),
          'Total', len(merge_dict(valid)))

    return (train, valid,
            np.array(list(map(len, train.values())))/len(merge_dict(train)))
