"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave

def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    imsave(save_path, img)

def saveImg(img, path):
    img = ((img + 1.) * (255./2)).astype(np.float)
    #img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
    result = scipy.misc.toimage(img, high=np.max(img), low=np.min(img))
    #img = (255.99*img).astype('uint8')
    imsave(path, result)

def readImg(img_path):
    img = np.array(scipy.misc.imread(img_path, mode='RGB').astype(np.float))
    #img = np.array(scipy.misc.imread(img_path, mode='L').astype(np.float))
    img = ((img - img.min()) / (255-img.min()))
    img = img * 2 - 1
    return img
    #return np.array(img)/127.5 - 1.


