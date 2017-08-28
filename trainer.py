from __future__ import print_function

import os
#import StringIO
from io import StringIO
import scipy.misc
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

import models as models_began
import models_content_encoder as models
from utils import save_image, context_encoder_loader, context_encoder_image_mask

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = models_began.nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw(image, data_format):
    if data_format == 'NHWC':
        new_image = models_began.nhwc_to_nchw(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image):
    image = image/127.5 - 1.
    return image

def denorm_img(norm):
    return tf.clip_by_value((norm + 1)*127.5, 0, 255)

def save_image_nchw(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):

    temp = tf.identity(tensor)
    tf.transpose(temp, [0, 2, 3, 1])

    return save_image(temp, filename, nrow, padding, normalize, scale_each)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

class Trainer(object):
    def __init__(self, config, data_loader, mask_loader):
        print("config=====================")
        print(config)
        print("config=====================")
        self.config = config
        self.data_loader = data_loader
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        init_g_lr = tf.assign(self.g_lr, tf.maximum(config.g_lr, config.g_lr), name='init_g_lr')
        init_d_lr = tf.assign(self.d_lr, tf.maximum(config.d_lr, config.d_lr), name='init_d_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.data_format = config.data_format

        _, height, width, self.channel = \
                models_began.get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.mask_outside, self.mask_center = context_encoder_image_mask(mask_loader)


        self.is_train = config.is_train

        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        print('self.model_dir: ', self.model_dir)
        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)
    
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
    
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)


        print('test mask value ========================')
        #print (self.mask_outside[np.in1d(self.mask_outside, [0, 1], invert=True).reshape(self.mask_outside.shape)].flatten())
        temp = self.mask_outside.eval(session=self.sess)
        print(np.histogram(temp))
        print('test mask value ========================')


        if (config.init_lr):
            g_lr, d_lr = self.sess.run([init_g_lr, init_d_lr])
            print('init lr ', g_lr, ' ', d_lr)

        # if not self.is_train:
        #     # dirty way to bypass graph finilization error
        #     g = tf.get_default_graph()
        #     g._finalized = False

        #     self.build_test_model()

    def train(self):
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))

        x_fixed = self.get_image_from_loader()

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_update": self.k_update,
                "measure": self.measure,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "k_t": self.k_t,
                    "g_lr": self.g_lr,
                    "d_lr": self.d_lr,
                })
            result = self.sess.run(fetch_dict)

            measure = result['measure']
            measure_history.append(measure)

            # test for mask loader
            # save_image_nchw(self.mask_outside, '{}/mask_outside.png'.format(self.model_dir))

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                g_lr   = result['g_lr']
                d_lr   = result['d_lr']
                k_t    = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} g_lr: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
                      format(step, self.max_step, d_loss, g_loss, g_lr, measure, k_t))

            if step % (self.log_step * 10) == 0:
                #x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])
                #cur_measure = np.mean(measure_history)
                #if cur_measure > prev_measure * 0.99:
                #prev_measure = cur_measure

    def build_model(self):
        self.x = self.data_loader
        self.x_nhwc = to_nhwc(self.x, self.data_format)
        x = norm_img(self.x)
        #print('x shape: ', x.get_shape().as_list())
        self.x_outside = context_encoder_loader(x, self.mask_outside)
        self.x_outside_nhwc = to_nhwc(self.x_outside, self.data_format)

        self.z = tf.random_uniform(
                (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        #G, self.G_var = models.GeneratorCNN(
        #        self.z, self.conv_hidden_num, self.channel,
        #        self.repeat_num, self.data_format, reuse=False)

        #d_out, self.D_z, self.D_var = models.DiscriminatorCNN(
        #        tf.concat([G, x], 0), self.channel, self.z_num, self.repeat_num,
        #        self.conv_hidden_num, self.data_format)
        #AE_G, AE_x = tf.split(d_out, 2)

        ## context encoder begin

        ##########################
        # this code should be tested and modified after trained enough - by snow
        batch_norm_is_train = True
        # batch_norm_is_train = self.is_train
        ##########################

        G, self.G_var = models.GeneratorCNN(
                self.x_outside, batch_norm_is_train, self.data_format, reuse=False)

        D_G, self.D_var = models.DiscriminatorCNN(
                G, batch_norm_is_train, self.data_format, reuse=False)

        D_x, self.D_var = models.DiscriminatorCNN(
                x, batch_norm_is_train, self.data_format, reuse=True)
        #D_all = tf.concat([D_x, D_G], 0)
        #print('D_all shape: ', D_all.get_shape().as_list())

        ## context encoder end

        self.G = denorm_img(G)
        self.G_nhwc = to_nhwc(self.G, self.data_format)
        self.composit_image_nhwc = to_nhwc(self.composit_images(self.x, self.G), self.data_format)
        self.D_G, self.D_x = D_G, D_x
        #self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            optimizer = tf.train.GradientDescentOptimizer
            # raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))
        g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)


        #self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
        #self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - G))

        #self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        #self.g_loss = tf.reduce_mean(tf.abs(AE_G - G))



        # loss - reconstruction
        loss_recon_ori = tf.square(x - G)
        # Loss for non-overlapping region
        loss_recon_center = tf.reduce_mean(
                tf.sqrt(1e-5 + tf.reduce_sum(loss_recon_ori * self.mask_center, [1,2,3])))
        # Loss for overlapping region
        self.loss_recon = loss_recon_center

        # 한번에 2개를 다 해야할수도 있음.
        d_lable = tf.concat([tf.ones([self.batch_size]), tf.zeros([self.batch_size])], 0)
        #g_lable = tf.ones([self.batch_size])
        #self.d_loss = tf.reduce_mean(
        #        tf.nn.sigmoid_cross_entropy_with_logits(tf.concat([AE_x, AE_G], 0), d_lable)

        self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(D_x, [-1]),
                                                        labels=tf.ones([self.batch_size])))
        self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(D_G, [-1]),
                                                        labels=tf.zeros([self.batch_size])))

        lambda_recon = 0.9
        lambda_adv = 0.1
        self.d_loss = (self.d_loss_fake + self.d_loss_real) * lambda_adv
        self.g_loss = self.d_loss_fake * lambda_adv + self.loss_recon * lambda_recon
        # weight decay 적용하기
        #weight_decay_rate =  0.00001
        #G_W = filter(lambda x: x.name.endswith('W:0'), self.G_var)
        #D_W = filter(lambda x: x.name.endswith('W:0'), self.D_var)
        #self.d_loss += weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), G_W)))
        #self.g_loss += weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), D_W)))

        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
        g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # for op in update_ops:
        #     print('update ops', op.name)
        update_ops = tf.group(*update_ops)

        with tf.control_dependencies([update_ops, d_optim, g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("original", self.x_nhwc),
            tf.summary.image("original_out", self.x_outside_nhwc),
            tf.summary.image("G", self.G_nhwc),
            tf.summary.image("composit", self.composit_image_nhwc),

            tf.summary.scalar("D_G", tf.reduce_mean(tf.reshape(self.D_G, [-1]))),
            tf.summary.scalar("D_x", tf.reduce_mean(tf.reshape(self.D_x, [-1]))),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])

    def build_test_model(self):
        #with tf.variable_scope("test") as vs:
        #    # Extra ops for interpolation
        #    z_optimizer = tf.train.AdamOptimizer(0.0001)

        #    self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
        #    self.z_r_update = tf.assign(self.z_r, self.z)

        #G_z_r, _ = models.GeneratorCNN(
        #        self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True)

        with tf.variable_scope("test") as vs:
            kk = 1
            #self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
            #self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.G, {self.z: inputs})
        if path is None and save:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image_nchw(x, path)
            print("[*] Samples saved: {}".format(path))
        return x

    #def autoencode(self, inputs, path, idx=None, x_fake=None):
    #    items = {
    #        'real': inputs,
    #        'fake': x_fake,
    #    }
    #    for key, img in items.items():
    #        if img is None:
    #            continue
    #        if img.shape[3] in [1, 3]:
    #            img = img.transpose([0, 3, 1, 2])

    #        x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
    #        x = self.sess.run(self.AE_x, {self.x: img})
    #        save_image(x, x_path)
    #        print("[*] Samples saved: {}".format(x_path))

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        items = {
            'real': inputs
        }
        for key, img in items.items():
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])

            black_img = np.zeros_like(img)
            x, x_out, G, composit = self.sess.run(
                    [self.x_nhwc, self.x_outside_nhwc, self.G_nhwc, self.composit_image_nhwc],
                    {self.x: img})
            save_image(G,        os.path.join(path, '{}_G.png'.format(idx)))
            save_image(x,        os.path.join(path, '{}_original.png'.format(idx)))
            save_image(x_out,    os.path.join(path, '{}_original_out.png'.format(idx)))
            save_image(composit, os.path.join(path, '{}_Composit.png'.format(idx)))


    def composit_images(self, img_out, img_inner):
        return (img_out * self.mask_outside) + (img_inner * (1 - self.mask_outside))

    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_x, {self.D_z: z})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size/2)

        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image_nchw(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image_nchw(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image_nchw(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)

    def test_context_encoder(self):
        root_path = "./"#self.model_dir

        all_G_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()
            self.autoencode(
                    real1_batch, self.model_dir,
                    idx="test{}_real1".format(step))
            self.autoencode(
                    real2_batch, self.model_dir,
                    idx="test{}_real2".format(step))


    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        # if self.data_format == 'NCHW':
        #     x = x.transpose([0, 2, 3, 1])
        return x
