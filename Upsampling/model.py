# -*- coding: utf-8 -*-
# xinhai liu

import tensorflow as tf
from Upsampling.spu_net import SPU_Net
from Common.ops import add_scalar_summary, add_hist_summary
from Common import model_utils
from Common import pc_util
from Common.loss_utils import pc_distance, get_plane_constraint_loss_minimize, get_projection_loss, get_plane_constraint_loss_minimize_gt, get_uniform_loss, get_uniform_loss_diy, get_repulsion_loss, discriminator_loss, generator_loss
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
import logging
import os
from tqdm import tqdm
from glob import glob
import math
from time import time
from termcolor import colored
import numpy as np
from tf_ops.grouping.tf_grouping import knn_point, group_point
from Common.pc_util import downsample_points, downsample_points_random
from decompose import decompose_points
from Upsampling.data_loader import Fetcher
from Common.ops import normalize_point_cloud
from tf_ops.nn_distance import tf_nndistance
from Upsampling.data_loader import get_single_pc_patches_geo_dist
from sklearn.manifold.isomap import Isomap
from Common.point_operation import guass_noise_point_cloud


class Model(object):
    def __init__(self, opts, sess):
        self.sess = sess
        self.opts = opts
        self.name = 'AE'

    def allocate_placeholders(self, patch_num_points):
        self.input_x = tf.placeholder(tf.float32, shape=[self.opts.batch_size, 64, 3])
        self.input_y = tf.placeholder(tf.float32, shape=[self.opts.batch_size, 64 * 4, 3])

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.pc_radius = tf.placeholder(tf.float32, shape=[self.opts.batch_size])

    def build_model(self):
        self.G = SPU_Net(self.opts, self.is_training, name='generator')

        # X -> Y
        self.G_y = self.G(self.input_x)

        alpha = 100.0
        gamma = 10.0
        # yita = 20.0
        beta = 0.01
        # sigma = 1.0

        self.dis_loss = alpha * pc_distance(self.G_y, self.input_y, dis_type='cd', radius=self.pc_radius)
        self.r_G = tf.reshape(self.G_y, [int(self.opts.batch_size / 4), self.opts.patch_size * 4, 3])
        # self.r_T = tf.reshape(self.input_x, [int(self.opts.batch_size / 4), 64*4, 3])

        # self.t_R = tf.concat([self.r_G, self.r_T], axis=1)
        self.uniform_loss = gamma * get_uniform_loss(self.r_G)
        self.projection = beta * get_projection_loss(self.G_y, k=5)

        self.pu_loss = self.dis_loss + self.uniform_loss + self.projection + tf.losses.get_regularization_loss()

        self.total_gen_loss = self.pu_loss

        self.setup_optimizer()
        self.summary_all()

        self.visualize_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
        self.visualize_titles = ['input_x', 'fake_y', 'real_y']

    def summary_all(self):
        # summary
        add_scalar_summary('loss/dis_loss', self.dis_loss, collection='gen')
        add_scalar_summary('loss/total_gen_loss', self.total_gen_loss, collection='gen')

        self.g_summary_op = tf.summary.merge_all('gen')
        # self.d_summary_op = tf.summary.merge_all('dis')

        self.visualize_x_titles = ['input_x', 'fake_y', 'real_y']
        self.visualize_x_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
        self.image_x_merged = tf.placeholder(tf.float32, shape=[None, 1500, 1500, 1])
        self.image_x_summary = tf.summary.image('Upsampling', self.image_x_merged, max_outputs=1)

    def setup_optimizer(self):
        learning_rate_g = tf.where(
            tf.greater_equal(self.global_step, self.opts.start_decay_step),
            tf.train.exponential_decay(self.opts.base_lr_g, self.global_step - self.opts.start_decay_step,
                                       self.opts.lr_decay_steps, self.opts.lr_decay_rate, staircase=True),
            self.opts.base_lr_g
        )
        learning_rate_g = tf.maximum(learning_rate_g, self.opts.lr_clip)
        add_scalar_summary('learning_rate/learning_rate_g', learning_rate_g, collection='gen')

        # create pre-generator ops
        gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

        with tf.control_dependencies(gen_update_ops):
            self.G_optimizers = tf.train.AdamOptimizer(learning_rate_g, beta1=self.opts.beta).minimize(
                self.total_gen_loss, var_list=gen_tvars,
                colocate_gradients_with_ops=True,
                global_step=self.global_step)

    def get_train_pred(self, x):
        x = x.astype(np.float32)
        before_gen = tf.convert_to_tensor(x)
        after_gen, _ = self.G(before_gen)
        return self.sess.run(after_gen, feed_dict={self.is_training: False})

    def cal_geodisec(self, points):
        geo_matrix = np.zeros([points.shape[0], points.shape[1], points.shape[1]], dtype=np.float32)
        for i in range(points.shape[0]):
            im = Isomap()
            im.fit(points[i])
            geo_matrix[i] = im.dist_matrix_
        return geo_matrix

    def train(self, fetcher_tpye='pugan'):
        if fetcher_tpye == 'pugan':
            fetcher = Fetcher(self.opts)
        elif fetcher_tpye == 'kitti':
            fetcher = KittiFetcher(self.opts)
        fetcher.start()

        log_dir = os.path.join(self.opts.log_dir, '64')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.allocate_placeholders(fetcher.x_num_points)
        self.build_model()

        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(max_to_keep=None)
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.LOG_FOUT = open(os.path.join(log_dir, 'log_train.txt'), 'w')

        with open(os.path.join(log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments

        if self.opts.restore:
            restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(log_dir)
            self.saver.restore(self.sess, checkpoint_path)
            self.LOG_FOUT = open(os.path.join(log_dir, 'log_train.txt'), 'a')

        start = time()
        step = self.sess.run(self.global_step)

        cur_total_gen_loss = np.inf
        for epoch in range(self.opts.training_epoch):
            logging.info('**** EPOCH %03d ****\t' % (epoch))
            whole_x = None
            whole_y = None
            whole_radius = None
            num_batches = fetcher.num_batches
            for batch_idx in range(num_batches):
                batch_input_x, batch_input_y, batch_radius = fetcher.fetch() #, batch_m
                if whole_x is None:
                    whole_x = batch_input_x.copy()
                    whole_y = batch_input_y.copy()
                    whole_radius = batch_radius
                else:
                    whole_x = np.concatenate((whole_x, batch_input_x), axis=0)
                    whole_y = np.concatenate((whole_y, batch_input_y), axis=0)
                    whole_radius = np.concatenate((whole_radius, batch_radius), axis=0)

                feed_dict = {self.input_x: batch_input_x,
                             self.input_y: batch_input_y,
                             self.pc_radius: batch_radius,
                             self.is_training: True}

                # Update G network
                for _ in range(self.opts.gen_update):
                    # get previously generated images
                    _, g_total_loss, summary = self.sess.run(
                        [self.G_optimizers, self.total_gen_loss, self.g_summary_op], feed_dict=feed_dict)
                    self.writer.add_summary(summary, step)

                if step % self.opts.steps_per_print == 0:
                    self.log_string('-----------EPOCH %d Step %d:-------------' % (epoch, step))
                    self.log_string('  G_loss   : {}'.format(g_total_loss))
                    self.log_string(' Time Cost : {}'.format(time() - start))
                    start = time()
                step += 1

            total_gen_loss = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.opts.batch_size
                end_idx = (batch_idx + 1) * self.opts.batch_size
                batch_input_x = whole_x[start_idx:end_idx, ...]
                batch_input_y = whole_y[start_idx:end_idx, ...]
                batch_radius = whole_radius[start_idx:end_idx]
                feed_dict = {self.input_x: batch_input_x,
                             self.input_y: batch_input_y,
                             self.pc_radius: batch_radius,
                             self.is_training: False}
                total_gen_loss += self.sess.run([self.total_gen_loss], feed_dict)[0]
            if epoch == int(self.opts.training_epoch * 0.3):
                cur_total_gen_loss = total_gen_loss
                if not self.opts.restore:
                    self.saver.save(self.sess, os.path.join(log_dir, 'model'), epoch)
                else:
                    self.saver.save(self.sess, os.path.join(log_dir, 'model'), epoch + 500)
                print(colored('Model saved at %s' % log_dir, 'white', 'on_blue'))
                # pred = self.get_train_pred(whole_x[:self.opts.batch_size])
            elif epoch > int(self.opts.training_epoch * 0.4) and cur_total_gen_loss > total_gen_loss:
                cur_total_gen_loss = total_gen_loss
                if not self.opts.restore:
                    self.saver.save(self.sess, os.path.join(log_dir, 'model'), epoch)
                else:
                    self.saver.save(self.sess, os.path.join(log_dir, 'model'), epoch + 500)
                print(colored('Model saved at %s' % log_dir, 'white', 'on_blue'))
                # pred = self.get_train_pred(whole_x[:self.opts.batch_size])

            # record loss
            """
            batch_input_x = whole_x[0: self.opts.batch_size, ...]
            batch_input_y = whole_y[0: self.opts.batch_size, ...]
            batch_radius = whole_radius[0: self.opts.batch_size, ...]
            feed_dict = {self.input_x: batch_input_x,
                         self.input_y: batch_input_y,
                         self.pc_radius: batch_radius,
                         self.is_training: False}
            dis_loss, plane_loss, uni_loss = self.sess.run([self.dis_loss, self.plane_constraint_loss,
                                                            self.uniform_loss], feed_dict)
            self.log_string('%%% dis: ' + str(dis_loss))
            self.log_string('%%% plane: ' + str(plane_loss))
            self.log_string('%%% uni: ' + str(uni_loss))
            """

    def get_patches(self, pc, test=False, patch_type='normal', filename=''):
        """
        :param pc: 1 * num_points * num_channels
        :return: num_patch * num_patch_point * num_channels
        """
        patch_num_points = self.opts.patch_size
        if test:
            seed1_num = self.opts.num_patch
        else:
            seed1_num = self.opts.num_patch

        if patch_type == 'normal':
            seed, seed_list = downsample_points(pc[0], seed1_num)
            patches = pc_util.extract_knn_patch(seed, pc[0, ...], patch_num_points)
        elif patch_type == 'geo':
            patches = get_single_pc_patches_geo_dist(filename, os.path.join('data', 'geo_dist'),
                                               self.opts.patch_size, seed1_num)
        # np.save('./data/vis_data/camel_patches_cd1.npy', patches)  # save patch points
        return patches

    def test(self):
        files = glob(os.path.join(self.opts.test_data_dir, '*.xyz'))
        flag = False
        for file in files:
            print('*********', file)
            start_time = time()
            pc = np.expand_dims(np.loadtxt(file), axis=0)
            # pc = guass_noise_point_cloud(pc, sigma=0.02)  # add guass noise
            # np.savetxt(os.path.join(self.opts.output_dir, 'input_%s' % os.path.basename(file)), pc[0], fmt='%.6f')
            num_patches = self.opts.num_patch
            num_patch_points = 256
            num_parts = 4
            num_part_points = int(num_patch_points / num_parts)

            pointsall = []

            if self.opts.use_geo:
                patches = self.get_patches(pc, test=True, patch_type='geo', filename=file)  # 24*256*3
            else:
                patches = self.get_patches(pc, test=True, patch_type='normal', filename=file)
            patches, centroid, furthest_distance = pc_util.normalize_point_cloud(patches)

            patches_part_pcs = np.zeros((num_patches * num_parts, num_part_points, 3))
            for i in range(num_patches):
                parts_pcs, parts_pcs_list = decompose_points(patches[i:i + 1], num_parts=4)
                patches_part_pcs[i * num_parts:(i + 1) * num_parts] = parts_pcs

            if flag == False:
                is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
                input_patches = tf.placeholder(tf.float32, shape=[patches_part_pcs.shape[0], patches_part_pcs.shape[1], 3])
                Gen = SPU_Net(self.opts, is_training, name='generator')
                pred_patches = Gen(input_patches)
                saver = tf.train.Saver()
                restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(
                    os.path.join(self.opts.log_dir, str(patches_part_pcs.shape[1])))
                print(colored('Model loaded at %s' % checkpoint_path, 'white', 'on_red'))
                saver.restore(self.sess, checkpoint_path)
                print('Model load!!!!!!')
                flag = True

            patches_parts = self.sess.run(pred_patches, feed_dict={input_patches: patches_part_pcs}) #, input_m: patches_part_geodesic #[0]
            patches_part_pcs = patches_parts.reshape((num_patches, num_parts * num_patch_points, 3))
            patches_part_pcs = patches_part_pcs * furthest_distance + centroid
            pointsall.append(patches_part_pcs)

            points = np.array(pointsall).reshape((-1, 3))
            points, _ = downsample_points(points, 8192)
            np.savetxt(os.path.join(self.opts.output_dir, os.path.basename(file)), points, fmt='%.6f')

            print('Cost time per file: ', time() - start_time)

    def log_string(self, msg):
        # global LOG_FOUT
        logging.info(msg)
        self.LOG_FOUT.write(msg + "\n")
        self.LOG_FOUT.flush()
