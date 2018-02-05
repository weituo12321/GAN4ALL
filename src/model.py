from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from six.moves import xrange

from tflib.save_images import *

class DAAE(object):
    def __init__(self, sess, image_size=28,
                 batch_size=64, z_dim=2*2*64,
                 c_dim=3,checkpoint_dir=None, lam=1e-3, leaky_factor = 0.2, M=10):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [3*3*64]
            c_dim: (optional) Dimension of image color. [3]
            lam: (optional) L1 lasso penalize parameter
            leaky_facor (optional) factor for leaky rely layer
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_shape = [image_size, image_size, c_dim]

        self.z_dim = z_dim

        self.lam = lam
        self.leaky_factor = leaky_factor
        self.M = M

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        #self.d_bn1 = batch_norm(name='d_bn1')
        #self.d_bn2 = batch_norm(name='d_bn2')
        #self.d_bn3 = batch_norm(name='d_bn3')

        #self.g_bn0 = batch_norm(name='g_bn0')
        #self.g_bn1 = batch_norm(name='g_bn1')
        #self.g_bn2 = batch_norm(name='g_bn2')
        #self.g_bn3 = batch_norm(name='g_bn3')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DAAE.model"

    def build_model(self):
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(
            tf.float32, [None] + self.image_shape, name='sample_images')

        self.z = tf.placeholder(tf.float32, [None] + [self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)


        self.corr_images = self.corrupt(self.images, mask_type='inpainting')
        self.En_corr = self.encoder(self.corr_images)
        self.De_corr = self.decoder(self.En_corr, self.c_dim)

        self.recon_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.De_corr, self.images))))

        #reuse the the encoder !!
        self.dis_out_real, self.dis_logits_real = self.discriminator(self.z)


        self.fake_z = tf.zeros_like(self.En_corr, name='fake_z')

        # how to add fake z many times ?
        for i in range(self.M):
            #self.fake_z += self.En_corr  # one solution ?

            self.corr_images = self.corrupt(self.images, mask_type='inpainting')   # implement corruption function
            self.fake_z += self.encoder(self.corr_images, reuse=True)

        self.fake_z /= self.M
        self.fake_z_sum = tf.summary.histogram("fake_z", self.fake_z)


        self.dis_out_fake, self.dis_logits_fake = self.discriminator(self.fake_z, reuse=True)

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dis_logits_fake,
                labels=tf.zeros_like(self.dis_out_fake)))

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dis_logits_real,
                labels=tf.ones_like(self.dis_out_real)))

        self.dis_loss = self.d_loss_fake + self.d_loss_real

        self.en_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.dis_logits_fake,
                labels=tf.ones_like(self.dis_out_fake)))


        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.dis_loss_sum = tf.summary.scalar("dis_loss", self.dis_loss)

        self.recon_loss_sum = tf.summary.scalar("recon_loss", self.recon_loss)
        self.en_loss_sum = tf.summary.scalar("en_loss", self.en_loss)

        t_vars = tf.trainable_variables()

        self.dis_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        self.de_vars = [var for var in t_vars if var.name.startswith('decoder')]
        self.en_vars = [var for var in t_vars if var.name.startswith('encoder')]
        self.ed_vars = self.de_vars + self.en_vars

        self.saver = tf.train.Saver(max_to_keep=1)

        # testing reconstruction
        self.recon = self.decoder(self.encoder(self.sample_images, reuse=True), self.c_dim, is_train=False)


    def corrupt(self, images, mask_type='inpainting',noise=False):
        image_shape = images.get_shape().as_list()[1:]
        mu = 0.0
        sigma = 0.1
        gnoise = np.random.normal(loc=mu, scale=sigma, size=image_shape)

        if mask_type == 'inpainting':
            a, b = 2, 2
            pixFrac = np.random.beta(a,b)
            mask = np.random.binomial(1, pixFrac, image_shape)
            result = tf.multiply(images, mask)

            if noise == True:
                return tf.add(result, gnoise)
            return result
        # TODO
        #if mask_type == 'compress':


    def sample_from_dis(self,dis = None):
        if dis == 'laplacian':
            mu = 0.0
            lam = 1.0
            return np.random.laplace(loc=mu, scale=lam, size=(self.batch_size, self.z_dim))
        elif dis == 'gaussian':
            mu = 0
            lower = -1
            upper = 1
            sigma = 1 / np.sqrt(self.z_dim)
            z = stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=(self.batch_size,self.z_dim))
            return z


    def train(self, config):
        data = glob(os.path.join(config.dataset, "*.jpg"))
        np.random.shuffle(data)
        assert(len(data) > 0)
        test_samples = glob(os.path.join(config.validation_set, "*.png"))
        test_original_samples = glob(os.path.join(config.original_validation_set, "*.jpg"))
        test_images = [readImg(batch_file).reshape(self.image_shape) for batch_file in test_samples]
        original_test_images = [readImg(batch_file).reshape(self.image_shape) for batch_file in test_original_samples]

        dis_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                          .minimize(self.dis_loss, var_list=self.dis_vars)
        ed_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                          .minimize(self.recon_loss, var_list=self.ed_vars)

        en_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                          .minimize(self.en_loss, var_list=self.en_vars)

        tf.global_variables_initializer().run()

        self.ed_sum = tf.summary.merge(
            [self.recon_loss_sum])
        self.en_sum = tf.summary.merge(
            [self.fake_z_sum,self.d_loss_fake_sum,self.en_loss_sum])
        self.dis_sum = tf.summary.merge(
            [self.z_sum,self.dis_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""

======
An existing model was found in the checkpoint directory.
If you want to train a new model from scratch,
delete the checkpoint directory or specify a different
--checkpoint_dir argument.
======

""")
        else:
            print("""

======
An existing model was not found in the checkpoint directory.
Initializing a new one.
======

""")

        for epoch in xrange(config.epoch):
            data = glob(os.path.join(config.dataset, "*.jpg"))
            assert(len(data) > 0)
            batch_idxs = len(data) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_images = [readImg(batch_file).reshape(self.image_shape) for batch_file in batch_files]
                # TODO
                #batch_corrs = self.corrupt(batch_images, 'inpainting')



                batch_z = self.sample_from_dis(dis='laplacian') # the true z from prior

                # Update ED network
                _, summary_str = self.sess.run([ed_optim, self.ed_sum],
                        feed_dict={ self.images: batch_images})
                self.writer.add_summary(summary_str, counter)

                # Update Discriminator network
                _, summary_str = self.sess.run([dis_optim, self.dis_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Update the encoder additionally
                _, summary_str = self.sess.run([en_optim, self.en_sum],
                        feed_dict={self.images: batch_images})
                self.writer.add_summary(summary_str, counter)


                errRecon = self.sess.run(self.recon_loss, feed_dict={self.images:batch_images})
                errDis = self.sess.run(self.dis_loss, feed_dict={self.images:batch_images, self.z:batch_z})


                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, dis_loss: %.8f, recon_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errDis, errRecon))

                if np.mod(counter, 100) == 1:
                    #samples, d_loss, g_loss = self.sess.run(
                    #    [self.sampler, self.d_loss, self.g_loss],
                    #    feed_dict={self.z: sample_z, self.images: sample_images}
                    #)
                    #save_images(samples, [8, 8],
                    #            './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    recon_images = self.sess.run(self.recon, feed_dict={self.sample_images:test_images})
                    #recon_images = self.sess.run(self.De_corr, feed_dict={self.images:original_test_images})
                    mse = ((recon_images - original_test_images) ** 2).mean()

                    print("[Sample_test] ########## reconstruction loss is %4.4f~" % mse)

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)


    def complete(self, config):
        os.makedirs(os.path.join(config.outDir, 'hats_imgs'), exist_ok=True)
        os.makedirs(os.path.join(config.outDir, 'completed'), exist_ok=True)

        tf.global_variables_initializer().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # data = glob(os.path.join(config.dataset, "*.png"))
        nImgs = len(config.imgs)
        batch_idxs = int(np.ceil(nImgs/config.batch_size))

        if config.maskType == 'inpainting':
            fraction_masked = 0.5
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'compress':
            mask = np.ones(self.image_shape)
            print("TODO")
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        else:
            assert(False)

        count = 0
        for idx in xrange(0, batch_idxs):
            batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
            batch_names = map(lambda x: x.split('/')[-1].split('.')[0], batch_files)
            batch_images = [readImg(batch_file) for batch_file in batch_files]

            if config.maskType == 'inpainting':
                hats_imgs = self.sess.run(self.corr_images, feed_dict={self.images:batch_images})
                recon_images = self.sess.run(self.recon, feed_dict={self.smaple_images:hats_imgs})

                for i in range(batch_images.shape[0]):
                    os.makedirs(os.path.join(config.outDir, 'hats_imgs'), exist_ok=True)
                    saveImg(hats_imgs[i], os.path.join(config.outDir, 'hats_imgs', "%05d.png" % count))
                    saveImg(samples[i], os.path.join(config.outDir, 'completed',"%05d.png" % count))
                    count += 1

    def discriminator(self, input_z, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            x1 = tf.layers.dense(input_z, 1000)
            x1 = tf.nn.relu(x1)

            x2 = tf.layers.dense(x1, 1000)
            x2 = tf.nn.relu(x2)

            logits = tf.layers.dense(x2, 1)
            out = tf.sigmoid(logits)

            return out, logits

    def encoder(self, images, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same')
            relu1 = tf.maximum(self.leaky_factor * x1, x1)
            # 14x14x64

            x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same')
            x2 = tf.layers.batch_normalization(x2, training=True)
            relu2 = tf.maximum(self.leaky_factor * x2, x2)
            # 7x7x128

            x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same')
            x3 = tf.layers.batch_normalization(x3, training=True)
            relu3 = tf.maximum(self.leaky_factor * x3, x3)
            # 4x4x256

            #x4 = tf.layers.conv2d(relu3, 256, 5, strides=2, padding='same')
            #bn4 = tf.layers.batch_normalization(x4, training=True)
            #relu4 = tf.maximum(0.2 * bn4, bn4)
            # 2 * 2 * 512

            # Flatten it
            flat = tf.contrib.layers.flatten(relu3)
            out = tf.layers.dense(flat, self.z_dim)

            return out

    def decoder(self, z, out_channel_dim, is_train=True):
        # we don't want to generate the image, but the negative noise!
        with tf.variable_scope("decoder", reuse= (not is_train)):
            x1 = tf.layers.dense(z, 4 * 4 * 256)
            # reshape it to start the conv stack
            x1 = tf.reshape(x1, [-1, 4, 4, 256])
            x1 = tf.layers.batch_normalization(x1, training=is_train)
            x1 = tf.maximum(self.leaky_factor * x1, x1)
            # 4 * 4 * 256

            x2 = tf.layers.conv2d_transpose(x1, 128, 4, strides=1, padding='valid')
            x2 = tf.layers.batch_normalization(x2, training=is_train)
            x2 = tf.maximum(self.leaky_factor * x2, x2)
            # 7 * 7 * 128

            x3 = tf.layers.conv2d_transpose(x2, 64, 5, strides=2, padding='same')
            x3 = tf.layers.batch_normalization(x3, training=is_train)
            x3 = tf.maximum(self.leaky_factor * x3, x3)
            # 14 * 14 * 64

            x4 = tf.layers.conv2d_transpose(x3, out_channel_dim, 5, strides=2, padding='same')
            #x4 = tf.layers.batch_normalization(x4, training=is_train)
            #x4 = tf.maximum(self.leaky_factor * x4, x4)
            # 24 * 24 * out_channel_dim


            # Output layer  out_chnnale_dim = K = 1 for residual nets
            #out = tf.layers.conv2d_transpose(x3, 1, 5, strides=1, padding='same')
            #out = tf.layers.batch_normalization(out, training=is_train)

            out = tf.tanh(x4)
            return out


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

