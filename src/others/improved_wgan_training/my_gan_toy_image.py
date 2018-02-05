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


#from __future__ import print_function

MODE = 'wgan-gp' # wgan or wgan-gp
DATASET = '8gaussians' # 8gaussians, 25gaussians, swissroll
DIM = 512 # Model dimensionality
FIXED_GENERATOR = False # whether to hold the generator fixed at real data plus
                        # Gaussian noise, as in the plots in the paper
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 200 # Batch size
ITERS = 10000 # how many generator iterations to train for
LEAKY_FACTOR = 0.01

lib.print_model_settings(locals().copy())

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output

def MyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output
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


def MyGenerator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_units, activation=None)
        #h1 = tf.layers.batch_normalization(h1) # for deep layers we need batch_norm
        h1 = tf.maximum(alpha*h1, h1)

        # logits and tanh output
        logits = tf.layers.dense(h1, out_dim)
        out = tf.tanh(logits)
        return out, logits


def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 192, DIM, inputs)
    output = ReLULayer('Discriminator.2', DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])

def MyDiscriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        #h1 = tf.layers.batch_normalization(h1) # for deep layers we need batch_norm
        h1 = tf.maximum(alpha*h1, h1)

        # logits and sigmoid output
        logits = tf.layers.dense(h1, 1)
        out = tf.sigmoid(logits)
        return out, logits

real_data = tf.placeholder(tf.float32, shape=[None, 192])
fake_data = Generator(BATCH_SIZE, real_data)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

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
    disc_interpolates = Discriminator(interpolates)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)

    disc_cost += LAMBDA*gradient_penalty

disc_params = lib.params_with_name('Discriminator')
gen_params = lib.params_with_name('Generator')

if MODE == 'wgan-gp':
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(
        disc_cost,
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(
            gen_cost,
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()

else:
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
        disc_cost,
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
            gen_cost,
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()


    # Build an op to do the weight clipping
    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var,
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

print "Generator params:"
for var in lib.params_with_name('Generator'):
    print "\t{}\t{}".format(var.name, var.get_shape())
print "Discriminator params:"
for var in lib.params_with_name('Discriminator'):
    print "\t{}\t{}".format(var.name, var.get_shape())

frame_index = [0]
def generate_image(true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    points = points.reshape((-1,2))
    samples, disc_map = session.run(
        [fake_data, disc_real],
        feed_dict={real_data:points}
    )
    disc_map = session.run(disc_real, feed_dict={real_data:points})

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x,y,disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+')
    plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+')

    plt.savefig('./gan_toy_images/'+'frame'+str(frame_index[0])+'.jpg')
    frame_index[0] += 1

# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':

        dataset = []
        for i in xrange(100000/25):
            for x in xrange(-2, 3):
                for y in xrange(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828 # stdev
        while True:
            for i in xrange(len(dataset)/BATCH_SIZE):
                yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5 # stdev plus a little
            yield data

    elif DATASET == '8gaussians':

        scale = 2.
        centers = [
            (1,0),
            (-1,0),
            (0,1),
            (0,-1),
            (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)),
            (-1./np.sqrt(2), 1./np.sqrt(2)),
            (-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        centers = [(scale*x,scale*y) for x,y in centers]
        while True:
            dataset = []
            for i in xrange(BATCH_SIZE):
                point = np.random.randn(2)*.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414 # stdev
            yield dataset

# Train loop!
input_dir = "../../data/faces101/training_patches"
checkDir = "./checkpoint"

output_dir = "./outputDir"
VEC_LEN = 192



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


saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as session:

    tf.global_variables_initializer().run()
    #session.run(tf.initialize_all_variables())
    gen = my_train_gen(input_dir)
    start_time = time.time()
    for iteration in xrange(ITERS):
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)
        # Train critic
        for i in xrange(CRITIC_ITERS):
            imgs = next(gen)
            _data = np.reshape(imgs, [-1, VEC_LEN])
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])


        if iteration % 10 == 0:
            errD = _disc_cost
            errG = gen_cost.eval({real_data:_data})
            errReal = tf.reduce_mean(disc_real).eval({real_data:_data})
            print("Iter: [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, d_real: %.8f" \
                    % (iteration, time.time() - start_time, errD, errG, errReal))

        if iteration % 500 == 0:
            saver.save(session, os.path.join(checkDir, "toy_image.model"), global_step=iteration)


        # Write logs and save samples
        lib.plot.plot('disc cost', _disc_cost)
        #if iteration % 100 == 0:
        #    lib.plot.flush()
        #    generate_image(_data)
        lib.plot.tick()

    """
    mu = 0
    lower = -1
    upper = 1
    sigma = 1 / np.sqrt(VEC_LEN)
    z = stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=(200,VEC_LEN))
    """
    sample = Generator(200, _data)
    genImgs = session.run(sample)
    for i in range(200):
        saveImg(np.reshape(genImgs[i], [8,8,3]), os.path.join(output_dir, "genImg%03d.png" % i))





