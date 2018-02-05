# python2

import os
import scipy
import numpy as np
import scipy.misc
from scipy.misc import imsave
from glob import glob


patch_places = "./"
patch_before = "../validation"

a, b = 2, 2
#frac = np.random.beta(a,b)

patch_size = (8,8)
data = glob(os.path.join(patch_before, "*.jpg"))
assert(len(data) > 0)

frac_list = list()
shape = (28,28,3)
w, h, c = shape
for i, img_path in enumerate(data):
    img = np.array(scipy.misc.imread(img_path, mode='RGB').astype(np.float))

    mask = np.ones(img.shape)
    frac = np.random.beta(a, b)
    frac_list.append(frac)
    mask[np.random.random((w, h)) < frac] = 0.0
    masked_img = np.multiply(img, mask)
    result = scipy.misc.toimage(masked_img, high=np.max(img), low=np.min(img))
    file_name = patch_places + "/image_%d.png" % (i)
    imsave(file_name, result)

with open('frac_list.txt', 'w') as writer:
    for item in frac_list:
        writer.write(str(item))
        writer.write('\n')
#scipy.misc.imsave("./corrupted.jpg", masked_img)

#patches = extract_patches_2d(img, patch_size, max_patches=8)

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
test_image = "./image001.jpg"
img = io.imread(test_image)
patches = extract_patches_2d(img, patch_size, max_patches=NUM)
for i in range(len(patches)):
    patch_idx = "%05d" % (i)
    patch_name = patch_places + "/" + patch_idx + ".jpg"
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

#mask = np.ones(img.shape)
#mask[np.random.random((w, h)) < frac] = 0.0
#masked_img = np.multiply(img, mask)
#scipy.misc.imsave("./corrupted.jpg", masked_img)

#patches = extract_patches_2d(img, patch_size, max_patches=8)
"""
for i in range(len(patches)):
    patch_idx = "%05d" % (i)
    patch_name = patch_before + "/" + patch_idx + ".png"
    patch = scipy.misc.toimage(patches[i], high=np.max(patches[i]), low=np.min(patches[i]))
    scipy.misc.imsave(patch_name, patch)

    mask = np.ones(list(patch_size) + [3])
    mask[np.random.random(patch_size) < frac] = 0

    masked_patch = np.multiply(mask, patches[i])
    patch_name = patch_places + "/" + patch_idx + ".png"
    masked_patch = scipy.misc.toimage(masked_patch, high=np.max(masked_patch), low=np.min(masked_patch))
    scipy.misc.imsave(patch_name, masked_patch)


sample1 = scipy.misc.imread(patch_before + "/" + "00000.png")
csample1 = scipy.misc.imread(patch_places + "/" + "00000.png")

print("##original ")
print(sample1[0,0,:])

print("#### corrupted")
print(csample1[0,0,:])
"""

"""
test_patch = patches[0]
scipy.misc.imsave("./test_original.png", test_patch)

new_patch = scipy.misc.imread("./test_original.png")
sample1 = new_patch
new_patch = np.array(new_patch) / 127.5 - 1.
result = (new_patch + 1.) * 127.5
result_img = scipy.misc.toimage(result, high=np.max(result), low=np.min(result))

scipy.misc.imsave("./temp.png", result_img)

csample1 = scipy.misc.imread("./temp.png")

print(test_patch[0,0,:])
print("##original ")
print(sample1[0,0,:])

print("#### corrupted")
print(csample1[0,0,:])
"""




