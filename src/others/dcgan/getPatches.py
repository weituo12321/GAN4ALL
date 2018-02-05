# python3

import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage import io

img_filenames = "./general/filenames"
#patch_places = "./general/patches"
patch_places = "./general/simple_patches"


patch_perc = 0.5
#patch_size = (64,64)
patch_size = (8,8)
NUM = 20000


"""
with open(img_filenames, 'r') as reader:
    imgfiles = reader.readlines()
    num_imgs = len(imgfiles)
    num_each = NUM // num_imgs
    patch_idx = 0
    for line in imgfiles:
        img_name = line.strip().split("/")[-1]
        print("addressing " + img_name)
        img = io.imread(line.strip())
        patches = extract_patches_2d(img, patch_size, max_patches=num_each)
        #patches = np.random.permutation(patches)
        #indices = np.random.permutation(np.arange(len(patches)))
        #for i in range(int(len(indices) * patch_perc)):
        for i in range(len(patches)):
            #idx = indices[i]
            #patch_idx = "_" + "%05d" % (idx)
            #patch_name = patch_places + "/" + img_name[:-4] + patch_idx + ".jpg"
            patch_name = patch_places + "/"+ "%04d" % (patch_idx) + ".jpg"
            patch_idx += 1
            io.imsave(patch_name, patches[i])

"""


"""
img = io.imread(test_image)
patches = extract_patches_2d(img, patch_size)
for i in range(len(patches)):
    patch_idx = "_" + "%05d" % (i)
    patch_name = patch_places + "image_0001"+ patch_idx + ".jpg"
    io.imsave(patch_name, patches[i])
"""


"""
patch_dir = patch_places + "*.jpg"
patches = np.array(io.imread_collection(patch_dir))
print(patches.shape)
img = reconstruct_from_patches_2d(patches, (334, 290, 3))
#print(img.shape)
#print(img[:,:,1])
#io.imsave("./reconstructed.jpg", img)
import scipy.misc
scipy.misc.imsave("./reconstructed.jpg", img)
"""

#img = io.imread("./general/simple_patches/" + "0000.jpg")
#img = io.imread("./101_ObjectCategories/Faces_easy/image_0435.jpg")
#img = io.imread("./lfw/aligned/George_W_Bush_0530.png")
for i in range(1000, 2001):
    img = io.imread("./general/simple_patches/%d.jpg" % (i))
    print(img[:,:,0])














