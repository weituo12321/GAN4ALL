# python3

import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage import io

img_filenames = "./101_ObjectCategories/Faces_easy_sub/filenames"
patch_places = "./101_ObjectCategories/Faces_easy_sub_patches/"
test_image = "./101_ObjectCategories/Faces_easy_sub/image_0001.jpg"

patch_size = (64,64)

"""
with open(img_filenames) as reader, open(patch_places) as writer:

    for line in reader.read_lines():
        img_name = line.split("/")[-1]
        print("addressing " + img_name)
        img = io.imread(line.strip())
        patches = extract_patches_2d(img, patch_size)
        for i in range(len(patches)):
            patch_idx = "_" + "%"
            patch_name = patch_places + "/" + img_name[:-4] +  + ".jpg"
            io.imsave(patches[i])

"""

"""
img = io.imread(test_image)
patches = extract_patches_2d(img, patch_size)
for i in range(len(patches)):
    patch_idx = "_" + "%05d" % (i)
    patch_name = patch_places + "image_0001"+ patch_idx + ".jpg"
    io.imsave(patch_name, patches[i])
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














