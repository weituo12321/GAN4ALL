
import numpy as np
import argparse
import os
from six.moves import xrange

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage import io
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('--patchSize', type=int, default=64)
parser.add_argument('--outDir', type=str, default='complete_apply')
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full'],
                    default='random')

parser.add_argument('imgs', type=str, nargs='+')

args = parser.parse_args()




def create_patches(config):

    for whole_img_path in config.imgs:
        img_name = whole_img_path.split("/")[-1].split(".")[0]
        whole_img = io.imread(whole_img_path)
        cur_img_shape = whole_img.shape

        #batch_idxs = int(np.ceil(nImgs/self.batch_size))
        if config.maskType == 'random':
            fraction_masked = 0.5
            #mask = np.ones(self.image_shape)
            mask = np.ones(cur_img_shape)
            #print(cur_img_shape)
            mask[np.random.random(cur_img_shape[0:2]) < fraction_masked] = 0.0
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        else:
            print("please use default maskType")
            assert(False)

        corrupted_img = np.multiply(whole_img, mask)

        corrupted_dir = os.path.join(config.outDir, 'corrupted_imgs')
        os.makedirs(corrupted_dir, exist_ok=True)
        corrupted_name = os.path.join(corrupted_dir, img_name + '_corrupted.jpg')
        scipy.misc.imsave(corrupted_name, corrupted_img)

        patches = extract_patches_2d(corrupted_img, (config.patchSize, config.patchSize))
        corrupted_patch_dir = os.path.join(config.outDir, img_name + '_corrupted_patches')
        os.makedirs(corrupted_patch_dir, exist_ok=True)
        for i in range(len(patches)):
            patch_idx = "%06d" % (i)
            patch_name = os.path.join(corrupted_patch_dir, patch_idx + ".jpg")
            scipy.misc.imsave(patch_name, patches[i])

create_patches(args)



