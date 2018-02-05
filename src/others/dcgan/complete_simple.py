#!/usr/bin/env python3.4
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import tensorflow as tf
import scipy.misc

from utils import *

from model import DCGAN_Simple

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions')
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full'],
                    default='full')
parser.add_argument('imgs', type=str, nargs='+')

args = parser.parse_args()

assert(os.path.exists(args.checkpointDir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    dcgan = DCGAN_Simple(sess, image_size=args.imgSize,
                  checkpoint_dir=args.checkpointDir, lam=args.lam)
    dcgan.complete(args)

    """
    check the learned parameters should be like dictionary elements
    """
    """
    ckpt = tf.train.get_checkpoint_state(args.checkpointDir)
    dcgan.saver.restore(sess, ckpt.model_checkpoint_path)

    dvars = dcgan.d_vars
    gvars = dcgan.g_vars

    d_w, d_b = dvars[0].value(), dvars[1].value()
    dw = sess.run(d_w)

    g_w, g_b = gvars[0].value(), gvars[1].value()
    gw = sess.run(g_w)

    num_gw = gw.shape[0] # currently we have 16
    # img size = 8 * 8 * 3
    elements = np.reshape(gw, (16, 8, 8, 3))
    scipy.misc.imsave("./weights_images/dw.jpg", np.reshape(dw, (8,8,3)))
    image_name = "./weights_images/gws.jpg"
    save_images(elements, [4,4], image_name)
    """













