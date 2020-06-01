import numpy as np
from skimage.exposure import rescale_intensity
import nibabel as nib
from utils import Summary

from tqdm import tqdm
from glob import glob
import nibabel as nib

if __name__=='__main__':
    s = Summary(port=39199, env='sample')
    fns = glob('data/ADNI/original/pp/*.nii')
    for fn in fns:
        sample = nib.load(fn)
        image = sample.get_data().astype(float)
        s.image3d('asdf', image)
