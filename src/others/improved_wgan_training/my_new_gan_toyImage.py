import os, sys
sys.path.append(os.getcwd())

import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.plot

from tflib.save_images import *
from glob import glob
import scipy.stats as stats
from itertools import cycle

MODE = 'wgan-gp' # wgan or wgan-gp
DATASET = '8gaussians' # 8gaussians, 25gaussians, swissroll
DIM = 512 # Model dimensionality
FIXED_GENERATOR = False # whether to hold the generator fixed at real data plus
                        # Gaussian noise, as in the plots in the paper
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 3 # How many critic iterations per generator iteration
BATCH_SIZE = 32 # Batch size
ITERS = 5000 # how many generator iterations to train for
LEAKY_FACTOR = 0.2
zdim = 100

lib.print_model_settings(locals().copy())


def MyGenerator(z, out_dim, n_units= 2 * 2* 128, reuse=False, alpha=0.2):
    with tf.variable_scope('Generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_units, activation=None)
        h1 = tf.reshape(h1, [-1, 2,2,128])
        h1 = tf.layers.batch_normalization(h1, training=True) # for deep layers we need batch_norm
        relu1 = tf.maximum(alpha*h1, h1)


        #h2 = tf.layers.dense(relu1, n_units *  2, activation=None)
        h2 = tf.layers.conv2d_transpose(relu1, 64, 3, strides = 2, padding='same')
        h2 = tf.layers.batch_normalization(h2, training=True) # for deep layers we need batch_norm
        relu2 = tf.maximum(alpha*h2, h2)

        # logits and tanh output
        #logits = tf.layers.dense(relu2, out_dim)
        logits = tf.layers.conv2d_transpose(relu2, out_dim, 3, strides = 2, padding='same')
        #print tf.get_shape(logits).as_list()
        logits = tf.contrib.layers.flatten(logits)
        out = tf.tanh(logits)
        return out, logits

def MyDiscriminator(x, n_units= 4 * 4 *64, reuse=False, alpha=0.2):
    with tf.variable_scope('Discriminator', reuse=reuse):
        #h1 = tf.layers.dense(x, n_units, activation=None)
        #imgnum = tf.get_shape(x).as_list()[0]
        x = tf.reshape(x, [-1, 8,8,3])
        h1 = tf.layers.conv2d(x, 64, 3, strides = 2, padding='same')
        #h1 = tf.layers.batch_normalization(h1, training=True) # for deep layers we need batch_norm
        relu1 = tf.maximum(alpha*h1, h1)

        #h2 = tf.layers.dense(relu1, n_units / 2, activation=None)
        h2 = tf.layers.conv2d(relu1, 128, 3, strides =2, padding= 'same')
        h2 = tf.layers.batch_normalization(h2,training=True) # for deep layers we need batch_norm
        relu2 = tf.maximum(alpha*h2, h2)
        # logits and sigmoid output

        flat = tf.contrib.layers.flatten(relu2)
        logits = tf.layers.dense(flat, 1)
        #logits = tf.layers.dense(logits, 1)
        out = tf.sigmoid(logits)
        return out, logits

tf.reset_default_graph()
real_data = tf.placeholder(tf.float32, shape=[None, 192], name='real_data')
input_z = tf.placeholder(tf.float32, shape=[None, zdim], name='input_z')
fake_data, glogits = MyGenerator(input_z, 3)

dout, disc_real = MyDiscriminator(real_data)
dout_g, disc_fake = MyDiscriminator(fake_data, reuse=True)

# WGAN loss
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN gradient penalty
if MODE == 'wgan-gp':
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1],
        minval=0.,
        maxval=1.
    )
    interpolates = alpha*real_data + ((1-alpha)*fake_data)
    _, disc_interpolates = MyDiscriminator(interpolates,reuse=True)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)

    disc_cost += LAMBDA*gradient_penalty

t_vars = tf.trainable_variables()
disc_params = [var for var in t_vars if var.name.startswith('Discriminator')]
gen_params = [var for var in t_vars if var.name.startswith('Generator')]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')):
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
        ).minimize(
            disc_cost,
            var_list=disc_params
        )

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')):
    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
        ).minimize(
            gen_cost,
            var_list=gen_params
        )



print "Generator params:"
for var in gen_params:
    print "\t{}\t{}".format(var.name, var.get_shape())
print "Discriminator params:"
for var in disc_params:
    print "\t{}\t{}".format(var.name, var.get_shape())


# Train loop!
#input_dir = "../../data/faces101/training_patches"
input_dir = "../../data/babara/patches"
checkDir = "./checkpoint"

output_dir = "./outputDir"
VEC_LEN = 192


# Dataset iterator
def my_train_gen(input_dir):
    data = glob(os.path.join(input_dir, "*.png"))
    assert(len(data) > 0)
    batch_idxs =len(data) // BATCH_SIZE
    result = list()
    for idx in xrange(0, batch_idxs):
        batch_files = data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
        batch = [readImg(batch_file) for batch_file in batch_files]
        result.append(batch)
    return cycle(result)

def my_noise_gen(batch_size, zdim):
    mu = 0
    lower = -1
    upper = 1
    sigma = 1 / np.sqrt(zdim)
    while True:
        z = stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=(batch_size,zdim))
        yield z



# for reconstruction part
#true_imgs_dir = "../../data/babara/trainingpatches_inp_original/"
#recon_input_dir = "../../data/babara/trainingpatches_inp_subset/"
#recon_output_dir = "./reconDir"
recon_input_dir = "../../data/babara/testInpainting/corrupted_patches"
recon_output_dir = "../../data/babara/testInpainting/recon_patches"
lam = 0.
zlen = zdim
ITER = 5000
momentum = 0.
lr = 0.001
leaky_factor = 0.2



data1 = glob(os.path.join(recon_input_dir, "*.png"))
assert(len(data1)) > 0
#completedata = glob(os.path.join(true_imgs_dir, "*.png"))
#assert(len(completedata)) > 0

names = map(lambda x: x.split('/')[-1].split('.')[0], data1)
realImgs = np.array([readImg(batch_file) for batch_file in data1])
#completeImgs = np.array([readImg(batch_file) for batch_file in completedata])

#mask = (realImgs != -1.0).astype(np.float32)

mask = tf.placeholder(tf.float32, [None,8,8,3])
real = tf.placeholder(tf.float32, [None,8,8,3])
contextual_loss = tf.reduce_mean(
        tf.reduce_sum(tf.abs(tf.multiply(mask, tf.reshape(fake_data, [-1, 8,8,3])) - tf.multiply(mask, real))))
perceptual_loss = -tf.reduce_mean(dout_g)
complete_loss = contextual_loss + lam * perceptual_loss
grad_complete_loss = tf.gradients(complete_loss, input_z)


saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as session:

    tf.global_variables_initializer().run()
    data_gen = my_train_gen(input_dir)
    noise_gen = my_noise_gen(BATCH_SIZE, zdim)
    start_time = time.time()
    for iteration in xrange(ITERS):
        # Train generator
        if iteration > 0:
            batchz = next(noise_gen)
            _ = session.run(gen_train_op, feed_dict={input_z:batchz})
        # Train critic
        for i in xrange(CRITIC_ITERS):
            imgs = next(data_gen)
            batchz = next(noise_gen)
            _data = np.reshape(imgs, [-1, VEC_LEN])
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data,input_z:batchz}
            )
            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])


        if iteration % 10 == 0:
            errD = _disc_cost
            errG = gen_cost.eval({input_z:batchz})
            errReal = tf.reduce_mean(disc_real).eval({real_data:_data})
            print("Iter: [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, d_real: %.8f" \
                    % (iteration, time.time() - start_time, errD, errG, errReal))

        if iteration % 50 == 0:
            saver.save(session, os.path.join(checkDir, "toy_image.model"), global_step=iteration)


        # Write logs and save samples
        lib.plot.plot('disc cost', _disc_cost)
        #if iteration % 100 == 0:
        #    lib.plot.flush()
        #    generate_image(_data)
        lib.plot.tick()


    samplez = next(noise_gen)
    genImgs = session.run(fake_data, feed_dict={input_z:samplez})
    for i in range(BATCH_SIZE):
        saveImg(np.reshape(genImgs[i], [8,8,3]), os.path.join(output_dir, "genImg%03d.png" % i))


    #complete process
    """
    complete_noise_gen = my_noise_gen(len(data1), zdim)
    zhats = next(complete_noise_gen)

    v = 0
    start_time = time.time()

    for i in range(ITER):
        runs = [complete_loss, grad_complete_loss]
        fd = {input_z: zhats}
        loss, g = session.run(runs, feed_dict = fd)

        prev = np.copy(v)
        v = momentum * v - lr * g[0]
        zhats += -momentum * prev + (1 + momentum) * v
        zhats = np.clip(zhats, -1, 1)

        if i % 50 == 0:

            tempsamples = fake_data.eval({input_z:zhats})
            tempsamples = np.reshape(tempsamples, [-1,8,8,3])
            #recover = np.multiply(mask, realImgs) + np.multiply(1 - mask, tempsamples)
            mse = np.mean((tempsamples - completeImgs)**2)

            print("Iter: [%4d] time: %4.4f, complete_loss: %.8f,mse: %.8f" % (i, time.time() - start_time, loss, mse))


    samples = fake_data.eval({input_z:zhats})
    samples = np.reshape(samples, [-1,8,8,3])
    #recover = np.multiply(mask, realImgs) + np.multiply(1 - mask, samples)
    for i in range(samples.shape[0]):
        #saveImg(recover[i], os.path.join(recon_output_dir, "recon%03d.png" % i))
        saveImg(samples[i], os.path.join(recon_output_dir, "recon%03d.png" % i))
    """


    # reconstruct the whole image
    recon_batch_size = 100
    data = glob(os.path.join(recon_input_dir, "*.png"))
    assert(len(data)) > 0
    names = map(lambda x: x.split('/')[-1].split('.')[0], data)

    complete_noise_gen = my_noise_gen(recon_batch_size, zdim)

    batch_idxs = len(data) // recon_batch_size
    start_time = time.time()
    for idx in xrange(0, batch_idxs):
        batch_files = data[idx * recon_batch_size:(idx+ 1) * recon_batch_size]
        patch_names = names[idx * recon_batch_size:(idx+ 1) * recon_batch_size]
        realImgs = np.array([readImg(batch_file) for batch_file in batch_files])

        curmask = (realImgs != -1.0).astype(np.float32)

        zhats = next(complete_noise_gen)
        v = 0

        for i in range(ITER):
            runs = [complete_loss, grad_complete_loss]
            fd = {real: realImgs, input_z: zhats, mask:curmask}
            loss, g = session.run(runs, feed_dict = fd)

            prev = np.copy(v)
            v = momentum * v - lr * g[0]
            zhats += -momentum * prev + (1 + momentum) * v
            zhats = np.clip(zhats, -1, 1)

            if i % 500 == 0:

                print("Iter: [%4d] time: %4.4f, complete_loss: %.8f" \
                        % (i, time.time() - start_time, loss))


        samples = fake_data.eval({input_z:zhats})
        samples = np.reshape(samples, [-1,8,8,3])
        for i in range(samples.shape[0]):
            saveImg(samples[i], os.path.join(recon_output_dir, patch_names[i] + ".png"))


