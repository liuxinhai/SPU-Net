import tensorflow as tf
from Upsampling.configs import FLAGS
from datetime import datetime
import os
import logging
import pprint
pp = pprint.PrettyPrinter()
import numpy as np
from Upsampling.data_loader import Fetcher
from time import time
from glob import glob
import sys

from Upsampling.model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


def run():
    start = time()

    # open session
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    # train
    if not FLAGS.restore:
        base_log_dir = FLAGS.log_dir
        FLAGS.log_dir = os.path.join(base_log_dir)
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
            os.makedirs(FLAGS.log_dir + '/geo_dist')
    else:
        base_log_dir = FLAGS.log_dir

    print('checkpoints:', FLAGS.log_dir)
    pp.pprint(FLAGS)

    # test
    FLAGS.output_dir = os.path.join(base_log_dir, 'output')
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    
    if FLAGS.train:
        code_dir = os.path.join(base_log_dir, 'code')
        if not os.path.exists(code_dir):
            os.mkdir(code_dir)
        os.system('cp *.py %s/' % code_dir)
        os.system('cp -r Common %s/' % code_dir)
        os.system('cp -r Upsampling %s/' % code_dir)

        with tf.Session(config=run_config) as sess:
            fetcher = Fetcher(FLAGS)
            total_steps = FLAGS.training_epoch * fetcher.num_batches
            FLAGS.start_decay_step = int(total_steps * 0.4)
            FLAGS.lr_decay_steps = total_steps * 4
            model = Model(FLAGS, sess)
            model.train()
        print('Train finished!')

    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        model = Model(FLAGS, sess)
        model.test()

    print('Test finished!')
    os._exit(0)


def main(unused_argv):
    run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
