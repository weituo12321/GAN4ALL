#!/usr/bin/env python3.4
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from six.moves import xrange
from model import DCGAN
from model import DCGAN_Simple
from ops import *
from utils import *
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions_apply')
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full'],
                    default='center')
#parser.add_argument('imgs', type=str, nargs='+')
parser.add_argument('imgs', type=str)
parser.add_argument('--batch_size', type=int, default='64')
parser.add_argument('--patch_size', type=int, default='64')

args = parser.parse_args()

assert(os.path.exists(args.checkpointDir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#os.makedirs(os.path.join(config.outDir, "apply_patches"))


def complete_apply(input_model, config):
    os.makedirs(os.path.join(config.outDir, 'complete_apply_recPatches'), exist_ok=True)

    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()

    isLoaded = input_model.load(config.checkpointDir)
    assert(isLoaded)

    # data = glob(os.path.join(config.dataset, "*.png"))
    data = glob(os.path.join(config.imgs, "*.jpg"))
    nImgs = len(data)

    batch_idxs = int(np.ceil(nImgs/args.batch_size))
    patchID = 0

    for idx in xrange(0, batch_idxs):
        print("Addressing the %05dth batch out of %d batches" % (idx, batch_idxs))
        l = idx*args.batch_size
        u = min((idx+1)*args.batch_size, nImgs)
        batchSz = u-l
        batch_files = data[l:u]
        batch = [get_image(batch_file, input_model.image_size, is_crop=input_model.is_crop) for batch_file in batch_files]
        batch_images = np.array(batch).astype(np.float32)
        #batch_mask = np.array(batch_images > 0, dtype=int)
        batch_mask = np.ones(batch_images.shape, dtype=int)
        #batch_mask = np.array(np.random.uniform(size=batch_images.shape) > 0.5, dtype=int)

        if batchSz < args.batch_size:
            print(batchSz)
            padSz = ((0, int(args.batch_size-batchSz)), (0,0), (0,0), (0,0))
            batch_images = np.pad(batch_images, padSz, 'constant')
            batch_images = batch_images.astype(np.float32)

            batch_mask = np.pad(batch_mask, padSz, "constant")
            batch_mask = batch_mask.astype(np.float32)

        zhats = np.random.uniform(-1, 1, size=(args.batch_size, input_model.z_dim))

        v = 0

        #masked_images = np.multiply(batch_images, batch_mask)
        masked_images = batch_images

        ### reconstruct current batch
        G_imgs = None
        my_contextual_loss = tf.reduce_mean(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(batch_mask, input_model.G) - tf.multiply(batch_mask, input_model.images))), 1)
        #my_contextual_loss = tf.nn.l2_loss(tf.multiply(batch_mask, input_model.G) - tf.multiply(batch_mask, input_model.images))
        my_grad = tf.gradients(my_contextual_loss, input_model.z)

        for i in xrange(config.nIter):
            fd = {
                input_model.z: zhats,
                input_model.mask: batch_mask,
                input_model.images: batch_images,
            }
            #run = [input_model.contextual_loss, tf.gradients(input_model.contextual_loss, input_model.z), input_model.G]
            #run = [input_model.complete_loss, input_model.grad_complete_loss, input_model.G]
            run = [my_contextual_loss, my_grad, input_model.G]
            loss, g, G_imgs = input_model.sess.run(run, feed_dict=fd)


            v_prev = np.copy(v)
            v = config.momentum*v - config.lr*g[0]
            zhats += -config.momentum * v_prev + (1+config.momentum)*v
            zhats = np.clip(zhats, -1, 1)
            #print(zhats[0,:])

            if i % 50 == 0:
                print(i, np.mean(loss[0:batchSz]))
                #print(i, loss)



        inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)

        completeed = masked_images + inv_masked_hat_images

        #if batchSz < config.batch_size:
        #    reconstruct_patches[l:u] = completeed[0:batchSz]
        #else:
        #    reconstruct_patches[l:u] = completeed

        # code for when batchSz smaller than the config.batch_size
        """
        if batchSz < config.batch_size:
            for reconstruct_patch in completeed[:batchSz, :,:,:]:
                patchName = os.path.join(config.outDir,'complete_apply_recPatches', 'patch_%06d.jpg' % patchID)
                img_save(patchName, reconstruct_patch)
        else:
            for reconstruct_patch in completeed:
                patchName = os.path.join(config.outDir,'complete_apply_recPatches', 'patch_%06d.jpg' % patchID)
                img_save(patchName, reconstruct_patch)

        """
        if batchSz < config.batch_size:
            genimg_name = os.path.join(config.outDir,'complete_apply_genPatches', 'genResult.jpg')
            save_images(G_imgs[:batchSz,:,:,:], [1, 8], genimg_name)

            oriimg_name = os.path.join(config.outDir,'complete_apply_genPatches', 'original.jpg')
            save_images(masked_images[:batchSz,:,:,:], [1, 8], oriimg_name)

            comp_name = os.path.join(config.outDir,'complete_apply_genPatches', 'complete.jpg')
            save_images(completeed[:batchSz,:,:,:], [1, 8], comp_name)


        patchID += 1





def create_patches(whole_img_path):
    os.makedirs(os.path.join(args.outDir, 'complete_apply_patches'), exist_ok=True)
    #size_list = list()
    #for whole_img_path in args.imgs:
    img_name = whole_img_path.split("/")[-1].split(".")[0]
    whole_img = imread(whole_img_path)
    cur_img_shape = whole_img.shape
    patches = get_patches(whole_img, args.patch_size)

    for i in range(len(patches)):
        patch_idx = "%06d" % (i)
        patch_places = os.path.join(args.outDir, 'complete_apply_patches', img_name)
        patch_name = patch_places + patch_idx + ".jpg"
        img_save(patch_name, patches[i])
    #size_list.append(cur_img_shape)
    #return size_list
    return cur_img_shape




with tf.Session(config=config) as sess:
    #dcgan = DCGAN(sess, image_size=args.imgSize,
    #              checkpoint_dir=args.checkpointDir, lam=args.lam)
    #dcgan.complete(args)
    dcgan = DCGAN_Simple(sess, image_size=args.imgSize,
                  checkpoint_dir=args.checkpointDir, lam=args.lam)
    complete_apply(dcgan, args)



# reconstruct the whole image
"""
print("Done Reconstruction on Patches, Start to reconstruct whole image!!!")
whole_img_dir = './data/general/apply'
whole_img_path = os.path.join(whole_img_dir, 'image_0101.jpg')
cur_img_shape = get_img_size(whole_img_path)

patch_dir = os.path.join(args.outDir, 'complete_apply_recPatches', '*.jpg')
patches = np.array(io.imread_collection(patch_dir))

reconstruct_img = reconstruct_from_patches_2d(patches, cur_img_shape)
imgName = os.path.join(config.outDir, 'complete_apply_recWhole', image_0101+'_reconstruct.jpg')
img_save(imgName, reconstruct_img)

print("Done")
"""
