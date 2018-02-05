#!/usr/bin/env python2.7

# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/main.py
#   + License: MIT
# [2017-08-15] Modifications for adversarial autoencoder:Weituo (http://weituo12321.github.com)
#   + License: MIT

import os
import scipy.misc
import numpy as np

from model import DAAE
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 2, "Epoch to train [2]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "First momentum term of adam [0.5]")
flags.DEFINE_float("beta2", 0.9, "Second momentum term of adam [0.9]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 28, "The size of image to use")
flags.DEFINE_string("dataset", "../data/mnist", "Dataset directory.")

flags.DEFINE_string("validation_set", "validation_corrupted", "Corrupted validation directory.")
flags.DEFINE_string("original_validation_set", "validation", "Original validation directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.validation_set):
    os.makedirs(FLAGS.validation_set)
if not os.path.exists(FLAGS.original_validation_set):
    os.makedirs(FLAGS.original_validation_set)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    daae= DAAE(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,c_dim=1,checkpoint_dir=FLAGS.checkpoint_dir)

    daae.train(FLAGS)
