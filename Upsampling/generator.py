# -*- coding: utf-8 -*-
# xinhai liu
import tensorflow as tf
from Common import ops
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from tf_ops.grouping.tf_grouping import knn_point,group_point

from Common.ops import attention_unit

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_CLIP = 0.99


class Generator(object):
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
            features = ops.feature_extraction(inputs, scope='feature_extraction', is_training=self.is_training,
                                              bn_decay=None)

            print(features.get_shape())
            print("here!!!!!")
            H = ops.pugan_up_projection_unit(features, self.up_ratio_real, scope="up_projection_unit",
                                       is_training=self.is_training, bn_decay=None)
            print(H.get_shape())
            print("thhere!!!!!")

            coord = ops.conv2d(H, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None)

            print(coord.get_shape())
            print("here11111!!!!!")

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])
            print(outputs.get_shape())
            print("here11111!!!!!")
            # input()
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return outputs
