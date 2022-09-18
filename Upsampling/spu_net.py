# -*- coding: utf-8 -*-
# xinhai liu
import tensorflow as tf
from Common import ops
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from tf_ops.grouping.tf_grouping import knn_point,group_point

from Common.ops import attention_unit
import numpy as np

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_CLIP = 0.99


class SPU_Net(object):
    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.up_ratio = self.opts.up_ratio
        # self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.up_ratio_real = self.up_ratio

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch * self.opts.batch_size,
            # BN_DECAY_DECAY_STEP,
            self.opts.lr_decay_steps,
            BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def __call__(self, inputs):
        """
        :param inputs: B*N*C
        :return: B*4N*C
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            features = ops.feature_extraction2(inputs, scope='feature_extraction', is_training=self.is_training, bn_decay=None)
            # print('11111111111111')
            # print(features.get_shape())

            with tf.variable_scope('up_unit', reuse=tf.AUTO_REUSE):
                L = ops.conv2d(features, 256, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=self.is_training, scope='conv0', bn_decay=None)

                num_point = L.get_shape()[1]
                grid = ops.gen_grid(2)
                # print(grid.get_shape())

                lgrid = tf.get_variable("grid1", shape=[num_point*4], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0), dtype=tf.float32)
                lgrid = tf.tile(tf.expand_dims(lgrid, 0), [tf.shape(L)[0], 1])

                grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(L)[0], 1, tf.shape(L)[1]])
                # print(grid.get_shape)

                grid = tf.reshape(grid, [tf.shape(L)[0], -1, 1, 2])
                lgrid = tf.reshape(lgrid, [tf.shape(L)[0], -1, 1, 2])

                H0 = ops.up_block_grid(L, tf.concat([grid, lgrid], axis=-1), is_training=self.is_training, bn_decay=None, scope='up_0') #tf.concat([grid, lgrid], axis=-1)

                # grid = tf.tile(grid, [1,2,1,1])
                grid1 = ops.gen_grid(2) # grid1 = tf.convert_to_tensor(np.array([[-0.2, -0.2], [0.2, -0.2]], dtype=np.float32))

                lgrid1 = tf.get_variable("grid2", shape=[num_point * 8], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0), dtype=tf.float32)
                lgrid1 = tf.tile(tf.expand_dims(lgrid1, 0), [tf.shape(L)[0], 1])

                ## grid1 = tf.convert_to_tensor(np.array([[2.0,4.0], [4.0, 0.0]], dtype=np.float32))

                grid1 = tf.tile(tf.expand_dims(grid1, 0), [tf.shape(L)[0], 1, 2*tf.shape(L)[1]])
                grid1 = tf.reshape(grid1, [tf.shape(L)[0], -1, 1, 2])
                # grid = tf.add(grid, grid1)
                lgrid1 = tf.reshape(lgrid1, [tf.shape(L)[0], -1, 1, 2])

                H = ops.up_block_grid(H0, tf.concat([grid1, lgrid1], axis=-1), is_training=self.is_training, bn_decay=None, scope='up_1') #tf.concat([grid, lgrid1], axis=-1)

                # print(H.get_shape())

            coord = ops.conv2d(H, 64, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=self.is_training, scope='fc_layer1', bn_decay=None)

            coord = ops.conv2d(coord, 3, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=self.is_training, scope='fc_layer3', bn_decay=None, activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])

            # print(outputs.get_shape())
            # input()
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return outputs
