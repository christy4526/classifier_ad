from skimage import exposure
from glob import glob
import os
import numpy as np
from PIL import Image
from visdom import Visdom

path = 'data/img/'
k = lambda x: os.path.split(x)[-1].split('.')[0]
filenames = sorted(glob(os.path.join(path, '*.jpg')),key=k)
vis = Visdom(port=39199, env='singer2')

for filename in filenames[:10]:
    image = np.array(Image.open(filename)).astype(float)/255.
    eq = exposure.equalize_hist(image)
    eq_clahe = exposure.equalize_adapthist(image,clip_limit=0.05)

    visimage = np.concatenate([
        image[np.newaxis].transpose(0,3,1,2),
        eq[np.newaxis].transpose(0,3,1,2),
        eq_clahe[np.newaxis].transpose(0,3,1,2)])
    vis.images(visimage, win=filename)
