

import os, sys
sys.path.append(os.getcwd())

import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.plot

from tflib.save_images import *
from glob import glob
import scipy.stats as stats
from itertools import cycle


def Generator(n_samples, real_data):
    if FIXED_GENERATOR:
        return real_data + (1.*tf.random_normal(tf.shape(real_data)))
    else:
        noise = tf.random_normal([n_samples, 48])
        output = ReLULayer('Generator.1', 48, 10, noise)
        #output = ReLULayer('Generator.2', DIM, DIM, output)
        #output = ReLULayer('Generator.3', DIM, DIM, output)
        output = lib.ops.linear.Linear('Generator.4', 10, 192, output)
        return tf.tanh(output)

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 192, DIM, inputs)
    output = ReLULayer('Discriminator.2', DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])


#input_dir = "../../data/faces101/testing_patches/original"
#input_dir = "../../data/faces101/testing_patches/fromtraining"
input_dir = "../../data/babara/trainingpatches_inp_subset/"
output_dir = "./reconDir"
lam = 0.1
zlen = 100
ITER = 5000
momentum = 0.
lr = 0.001
leaky_factor = 0.01

with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./checkpoint/toy_image.model-9950.meta")
    saver.restore(sess, "./checkpoint/toy_image.model-9950")
    model_graph = sess.graph
    #all_keys =  model_graph.get_all_collection_keys()
    trainable_vars =  model_graph.get_collection('trainable_variables')
    print trainable_vars

    """

    gw1, gb1 = trainable_vars[:2]
    gw2, gb2 = trainable_vars[2:4]

    dw1, db1 = trainable_vars[4:6]
    dw2, db2 = trainable_vars[6:8]
    #dw3, db3 = trainable_vars[8:10]
    #dw4, db4 = trainable_vars[10:12]

    data = glob(os.path.join(input_dir, "*.png"))
    assert(len(data)) > 0
    names = map(lambda x: x.split('/')[-1].split('.')[0], data)
    realImgs = np.array([readImg(batch_file) for batch_file in data])

    mask = (realImgs > -0.99).astype(np.float32)
    #mask = np.ones(realImgs.shape).astype(np.float32)

    z = tf.placeholder(tf.float32, [len(data), zlen])
    gout = tf.add(tf.matmul(z, gw1), gb1)
    gout = tf.maximum(leaky_factor * gout,gout)
    gout = tf.tanh(tf.add(tf.matmul(gout, gw2), gb2))
    gen_imgs = tf.reshape(gout, [-1, 8,8,3])

    dout = tf.add(tf.matmul(gout, dw1), db1)
    dout = tf.maximum(leaky_factor * dout, dout)
    dout = tf.add(tf.matmul(dout, dw2), db2)
    #dout = tf.nn.relu(tf.add(tf.matmul(dout, dw3), db3))
    #dout = tf.add(tf.matmul(dout, dw4), db4)

    # compute the real score
    real = tf.placeholder(tf.float32, [len(data), 8,8,3])
    _real = tf.contrib.layers.flatten(real)
    rdout = tf.add(tf.matmul(_real, dw1), db1)
    rdout = tf.maximum(leaky_factor * rdout, rdout)
    rdout = tf.add(tf.matmul(rdout, dw2), db2)
    #rdout = tf.nn.relu(tf.add(tf.matmul(rdout, dw3), db3))
    #rdout = tf.add(tf.matmul(rdout, dw4), db4)
    real_score = tf.reduce_mean(rdout)
    realscore = sess.run(real_score, feed_dict={real: realImgs})

    fake_score = tf.reduce_mean(dout)
    contextual_loss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(tf.multiply(mask, gen_imgs) - realImgs)))
    perceptual_loss = -tf.reduce_mean(dout)
    complete_loss = contextual_loss + lam * perceptual_loss
    grad_complete_loss = tf.gradients(complete_loss, z)

    #zhats = np.random.randn(len(data) * zlen).reshape(len(data), zlen)

    mu = 0
    lower = -1
    upper = 1
    sigma = 1 / np.sqrt(zlen)
    zhats = stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=(len(data),zlen))

    v = 0
    start_time = time.time()
    for i in range(ITER):
        runs = [complete_loss, grad_complete_loss,fake_score]
        fd = {z: zhats}
        loss, g, fakescore = sess.run(runs, feed_dict = fd)

        prev = np.copy(v)
        v = momentum * v - lr * g[0]
        zhats += -momentum * prev + (1 + momentum) * v
        zhats = np.clip(zhats, -1, 1)




        if i % 50 == 0:
            #print g[0][0]
            #print np.sum(np.abs(zhats))

            tempsamples = gen_imgs.eval({z:zhats})
            mse = np.mean((tempsamples - realImgs)**2)

            print("Iter: [%4d] time: %4.4f, complete_loss: %.8f, fake_score: %.8f, real_score: %.8f, mse: %.8f" \
                    % (i, time.time() - start_time, loss, fakescore, realscore,mse))


    samples = gen_imgs.eval({z:zhats})
    recover = realImgs
    recover[realImgs == -1] = samples[realImgs == -1]
    for i in range(samples.shape[0]):
        saveImg(recover[i], os.path.join(output_dir, "recon%03d.png" % i))
    """











