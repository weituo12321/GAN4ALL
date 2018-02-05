

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


def MyGenerator(z, out_dim, n_units= 2 * 2* 128, reuse=False, alpha=0.2):
    with tf.variable_scope('Generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_units, activation=None)
        h1 = tf.layers.batch_normalization(h1) # for deep layers we need batch_norm
        relu1 = tf.maximum(alpha*h1, h1)

        h2 = tf.layers.dense(relu1, n_units *  2, activation=None)
        bn2 = tf.layers.batch_normalization(h2) # for deep layers we need batch_norm
        relu2 = tf.maximum(alpha*bn2, bn2)

        # logits and tanh output
        logits = tf.layers.dense(relu2, out_dim)
        out = tf.tanh(logits)
        return out, logits

def MyDiscriminator(x, n_units= 4 * 4 *64, reuse=False, alpha=0.2):
    with tf.variable_scope('Discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        h1 = tf.layers.batch_normalization(h1) # for deep layers we need batch_norm
        relu1 = tf.maximum(alpha*h1, h1)

        h2 = tf.layers.dense(relu1, n_units / 2, activation=None)
        h2 = tf.layers.batch_normalization(h2) # for deep layers we need batch_norm
        relu2 = tf.maximum(alpha*h2, h2)
        # logits and sigmoid output
        logits = tf.layers.dense(relu2, 1)
        #logits = tf.layers.dense(logits, 1)
        out = tf.sigmoid(logits)
        return out, logits




#input_dir = "../../data/faces101/testing_patches/original"
#input_dir = "../../data/faces101/testing_patches/fromtraining"
true_imgs_dir = "../../data/babara/trainingpatches_inp_original/"
input_dir = "../../data/babara/trainingpatches_inp_subset/"
output_dir = "./reconDir"
lam = 0.
zlen = 100
ITER = 1000
momentum = 0.
lr = 0.001
leaky_factor = 0.2


data = glob(os.path.join(input_dir, "*.png"))
assert(len(data)) > 0
completedata = glob(os.path.join(true_imgs_dir, "*.png"))
assert(len(completedata)) > 0

names = map(lambda x: x.split('/')[-1].split('.')[0], data)
realImgs = np.array([readImg(batch_file) for batch_file in data])
completeImgs = np.array([readImg(batch_file) for batch_file in completedata])

mask = (realImgs != -1.0).astype(np.float32)
invermask = 1 - mask

temp = np.multiply(invermask, np.ones(mask.shape))
new = temp + realImgs
print np.min(new)
print np.max(new)
#print len(realImgs[realImgs == -1.0])
#print np.min(realImgs)
#print realImgs.shape
#print np.count_nonzero(mask)





"""
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./checkpoint/toy_image.model-9950.meta")
    saver.restore(sess, "./checkpoint/toy_image.model-9950")
    model_graph = sess.graph
    #all_keys =  model_graph.get_all_collection_keys()
    trainable_vars =  model_graph.get_collection('trainable_variables')
    #print trainable_vars


    gw1, gb1 = trainable_vars[:2]
    gbeta1,ggamma1 = trainable_vars[2:4]

    gw2, gb2 = trainable_vars[4:6]
    gbeta2, ggamma2 = trainable_vars[6:8]
    gw3, gb3 = trainable_vars[8:10]

    dw1, db1 = trainable_vars[10:12]
    dbeta1, dgamma1 = trainable_vars[12:14]
    dw2, db2 = trainable_vars[14:16]
    dbeta2, dgamma2 = trainable_vars[16:18]
    dw3, db3 = trainable_vars[18:20]

    data = glob(os.path.join(input_dir, "*.png"))
    assert(len(data)) > 0
    completedata = glob(os.path.join(true_imgs_dir, "*.png"))
    assert(len(completedata)) > 0

    names = map(lambda x: x.split('/')[-1].split('.')[0], data)
    realImgs = np.array([readImg(batch_file) for batch_file in data])
    completeImgs = np.array([readImg(batch_file) for batch_file in completedata])

    mask = (realImgs > -0.99).astype(np.float32)
    #mask = np.ones(realImgs.shape).astype(np.float32)

    z = tf.placeholder(tf.float32, [None, zlen])
    gout = tf.add(tf.matmul(z, gw1), gb1)
    curmean1, curvar1 = tf.nn.moments(gout,axes=[0])
    gout = (gout - curmean1) / tf.sqrt(curvar1 + 0.001)
    gout = ggamma1 * gout + gbeta1

    gout = tf.maximum(leaky_factor * gout,gout)

    gout = tf.add(tf.matmul(gout, gw2), gb2)
    curmean2, curvar2 = tf.nn.moments(gout,axes=[0])
    gout = (gout - curmean2) / tf.sqrt(curvar2 + 0.001)
    gout = ggamma2 * gout + gbeta2
    #gout = tf.layers.batch_normalization(gout, beta_initializer=gbeta, gamma_initializer=ggamma)

    gout = tf.maximum(leaky_factor * gout,gout)

    gout = tf.tanh(tf.add(tf.matmul(gout, gw3), gb3))
    gen_imgs = tf.reshape(gout, [-1, 8,8,3])

    dout = tf.add(tf.matmul(gout, dw1), db1)
    dcurmean1, dcurvar1 = tf.nn.moments(dout,axes=[0])
    dout = (dout - dcurmean1) / tf.sqrt(dcurvar1 + 0.001)
    dout = dgamma1 * dout + dbeta1

    dout = tf.maximum(leaky_factor * dout, dout)

    dout = tf.add(tf.matmul(dout, dw2), db2)
    dcurmean2, dcurvar2 = tf.nn.moments(dout,axes=[0])
    dout = (dout - dcurmean2) / tf.sqrt(dcurvar2 + 0.001)
    dout = dgamma2 * dout + dbeta2

    #dout = tf.layers.batch_normalization(dout, beta_initializer=dbeta, gamma_initializer=dgamma)
    dout = tf.maximum(leaky_factor * dout, dout)
    dout = tf.add(tf.matmul(dout, dw3), db3)
    dout_output = tf.sigmoid(dout)

    # compute the real score
    real = tf.placeholder(tf.float32, [None, 8,8,3])
    _real = tf.contrib.layers.flatten(real)
    rout = tf.add(tf.matmul(_real, dw1), db1)
    rcurmean1, rcurvar1 = tf.nn.moments(rout,axes=[0])
    rout = (rout - rcurmean1) / tf.sqrt(rcurvar1 + 0.001)
    rout = dgamma1 * rout + dbeta1

    rout = tf.maximum(leaky_factor * rout, rout)
    rout = tf.add(tf.matmul(rout, dw2), db2)
    rcurmean2, rcurvar2 = tf.nn.moments(rout,axes=[0])
    rout = (rout - rcurmean2) / tf.sqrt(rcurvar2 + 0.001)
    rout = dgamma2 * rout + dbeta2
    #rout = tf.layers.batch_normalization(rout, beta_initializer=dbeta, gamma_initializer=dgamma)

    rout = tf.maximum(leaky_factor * rout, rout)
    rout = tf.add(tf.matmul(rout, dw3), db3)
    #rout_output = tf.sigmoid(rout)
    real_score = tf.reduce_mean(rout)
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
            recover = realImgs
            recover[realImgs == -1] = tempsamples[realImgs == -1]
            mse = np.mean((recover - completeImgs)**2)

            print("Iter: [%4d] time: %4.4f, complete_loss: %.8f, fake_score: %.8f, real_score: %.8f, mse: %.8f" \
                    % (i, time.time() - start_time, loss, fakescore, realscore,mse))


    samples = gen_imgs.eval({z:zhats})
    recover = realImgs
    recover[realImgs == -1] = samples[realImgs == -1]
    for i in range(samples.shape[0]):
        saveImg(recover[i], os.path.join(output_dir, "recon%03d.png" % i))

"""










