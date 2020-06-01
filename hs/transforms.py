from __future__ import absolute_import, division, print_function

import os
from os.path import join as pjoin
from glob import glob
import random

from multiprocessing import Pool
from scipy.ndimage.interpolation import rotate, zoom
from skimage.exposure import rescale_intensity

import numpy as np
import nibabel as nib

from tqdm import tqdm

import torch
from torchvision.transforms import Compose, Lambda


class ToWorldCoordinateSystem(object):
    def __call__(self, image):
        image = np.rot90(image.T, 2)
        return image


class Normalize(object):
    def __init__(self, mean=0, std=1):
        # mean = 0.36 std = 0.33
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = (image-self.mean)/self.std
        return image


class ToFloatTensor(object):
    def __call__(self, image):
        image = torch.from_numpy(image.copy()).float().unsqueeze(0)
        image /= 255
        return image


class Flip(object):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, image):
        return np.flip(image, self.axis)


class AllFlip(object):
    def __call__(self, image):
        images = []
        for axis in range(3):
            images += [np.flip(image, axis).copy()]

        return images


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if (np.array(image.shape) == np.array(self.size)).all():
            return image

        s = np.array(self.size)
        r = np.array(image.shape) - s
        m = (r/2).astype(int)
        mr = np.array(image.shape)-s-m

        return image[m[0]:image.shape[0]-mr[0],
                     m[1]:image.shape[1]-mr[1],
                     m[2]:image.shape[2]-mr[2]]


class PartialNineCrop(object):
    def __init__(self, inshape, outshape, index):
        self.inshape = np.array(inshape)
        self.outshape = np.array(outshape)

        ins = self.inshape
        ots = self.outshape
        rem = np.array(ins) - np.array(ots)
        m = (rem/2).astype(int)
        mr = ots + m
        self.index = index
        self.indices = [
            (slice(0, ots[0]),    slice(0, ots[1]),    slice(0, ots[2])),
            (slice(0, ots[0]),    slice(rem[1], None), slice(0, ots[2])),
            (slice(0, ots[0]),    slice(0, ots[1]),    slice(rem[2], None)),
            (slice(0, ots[0]),    slice(rem[1], None), slice(rem[2], None)),

            (slice(rem[0], None), slice(0, ots[1]),    slice(0, ots[2])),
            (slice(rem[0], None), slice(rem[1], None), slice(0, ots[2])),
            (slice(rem[0], None), slice(0, ots[1]),    slice(rem[2], None)),
            (slice(rem[0], None), slice(rem[1], None), slice(rem[2], None)),

            (slice(m[0], mr[0]), slice(m[1], mr[1]), slice(m[2], mr[2]))
        ]

    def __call__(self, image):
        return image[self.indices[self.index]]


class NineCrop(PartialNineCrop):
    def __init__(self, inshape, outshape):
        super(NineCrop, self).__init__(inshape, outshape, None)

    def __call__(self, image):
        patches = []
        for i in range(9):
            self.index = i
            patches += [super(NineCrop, self).__call__(image)]
        return patches


class NineCropFlip(NineCrop):
    """
    get each corner and center cropped image patchs and their flip(all axis)
    """

    def __init__(self, size, axis):
        raise NotImplementedError
        super(NineCropFlip, self).__init__(size)
        self.flip = Flip(axis)

    def __call__(self, image):
        images = super(NineCropFlip, self).__call__(image)

        flipped = []
        for cim in images:
            flipped += [self.flip(cim)]
        return images+flipped


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if (np.array(image.shape) == np.array(self.size)).all():
            return image
        idcs = [random.choice(list(range(0, image.shape[i]-self.size[i]-1)))
                for i in range(len(self.size))]
        return image[idcs[0]:idcs[0]+self.size[0],
                     idcs[1]:idcs[1]+self.size[1],
                     idcs[2]:idcs[2]+self.size[2]]


class RandomFlip(object):
    def __init__(self, axis, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return np.flip(image, self.axis)
        return image


class RandomInverse(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return image.max() - image
        return image


class Resize(object):
    def __init__(self, shape):
        self.shape = np.array(shape)

    def __call__(self, image):
        im_shape = np.array(image.shape)
        return zoom(image, self.shape/im_shape)


class Stack(object):
    def __call__(self, images):
        return torch.stack(images)


def transform_presets(mode):
    transformer = []
    if mode == 'no augmentation':
        transformer += [ToFloatTensor()]
    elif mode == 'random crop':
        transformer += [RandomCrop((112, 144, 112))]
        transformer += [ToFloatTensor()]
    elif mode == 'nine crop':
        transformer += [NineCrop((132, 173, 135), (112, 144, 112))]
        transformer += [Lambda(lambda patches: torch.stack([
            ToFloatTensor()(patch) for patch in patches]))]

    transformer += [Normalize(0.5, 0.5)]
    return Compose(transformer)
