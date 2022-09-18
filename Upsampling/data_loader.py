# -*- coding: utf-8 -*-

import numpy as np
import os
from Common.pc_util import downsample_points, downsample_points_random, downsample_points_fps
from glob import glob
from Common import pc_util
from Common import point_operation
import queue
import threading
import scipy.io as scio
from sklearn.neighbors import NearestNeighbors
from decompose import decompose_points
import h5py


def normalize_point_cloud(input):
    if len(input.shape)==2:
        axis = 0
    elif len(input.shape)==3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance


def load_files(files):
    num_points = np.loadtxt(files[0]).shape[0]
    num_channels = np.loadtxt(files[0]).shape[1]
    pcs = np.zeros((len(files), num_points, num_channels))
    for i in range(len(files)):
        pcs[i, :, :] = np.loadtxt(files[i])

    return pcs


def load_h5_data(h5_filename='', opts=None, skip_rate = 1, use_randominput=True):
    num_point = 256
    num_4X_point = int(256*4)
    num_out_point = int(256*4)

    print("h5_filename : ",h5_filename)
    if use_randominput:
        print("use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_%d'%num_4X_point][:]
        gt = f['poisson_%d'%num_out_point][:]
    else:
        print("Do not randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_%d' % num_point][:]
        gt = f['poisson_%d' % num_out_point][:]

    #name = f['name'][:]
    assert len(input) == len(gt)

    print("Normalization the data")
    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    print("total %d samples" % (len(input)))
    return input, gt, data_radius


def downsample_pcs(pcs, downsample_num_points):
    down_pcs = np.zeros((pcs.shape[0], downsample_num_points, pcs.shape[2]))
    for i in range(len(pcs)):
        down_pcs[i, :, :] = downsample_points(pcs[i], downsample_num_points)

    return down_pcs


def get_single_pc_patches(pc, patch_size=256, num_patch=24):
    """
    :param pc: num_points * num_channels
    :return: num_patch * num_patch_point * num_channels
    """
    patch_num_points = patch_size

    seed1_num = num_patch
    points = np.expand_dims(pc, 0)

    seed, _ = downsample_points(points[0], seed1_num)
    patches = pc_util_lxh.extract_knn_patch(seed, points[0, ...], patch_num_points)
    return patches


def get_single_pc_patches_geo_dist(file, geo_dist_mat_dir, patch_size=256, num_patch=24):
    patch_num_points = patch_size

    patch_num_channels = np.loadtxt(file).shape[1]
    num_patches_per_pc = num_patch

    data = np.load(os.path.join(geo_dist_mat_dir, os.path.basename(file) + '_k5.npy')) #geo_dist_mat_dir ./data/geo_dist_4096
    points = np.loadtxt(file).astype(np.float32)  # 2048*3
    down_points, seed_list = pc_util.downsample_points(points, num_patches_per_pc)  # 24*3 center points

    # print("here!!!!!!!!!!!!!!!!!!!!!!")
    # np.save('./data/vis_data/camel_center_cd1.npy', down_points)  # save center points
    # np.save('./data/vis_data/camel_list_cd1.npy', seed_list)  # save center points
    knn_search = NearestNeighbors(n_neighbors=1, algorithm='auto')
    knn_search.fit(points)
    knn_idx = knn_search.kneighbors(down_points, return_distance=False)
    idx = np.squeeze(np.array(knn_idx), axis=1)

    # points = np.loadtxt(file)
    patches = np.zeros((num_patches_per_pc, patch_num_points, patch_num_channels))
    patches_list = np.zeros((num_patches_per_pc, patch_num_points), dtype=np.int32)

    for i in range(idx.shape[0]):
        # print(np.argsort(data[idx[i]])[:patch_num_points].shape)
        # input()
        points_list = np.argsort(data[idx[i]])[:patch_num_points]
        patches_list[i, ...] = points_list
        patches[i, ...] = points[points_list]
    # np.save('./data/vis_data/camel_patch_list_cd1.npy', patches_list)  # save center points
    return patches


def prepare_train_data(files, patch_size=256, num_patch=24, patch_type='normal'):
    # pcs = load_files(files)
    pc = np.loadtxt(files[0])
    patch_num_points = patch_size

    patch_num_channels = pc.shape[1]
    num_patches_per_pc = num_patch
    patches = np.zeros((len(files) * num_patches_per_pc, patch_num_points, patch_num_channels))
    if patch_type == 'normal':
        for i in range(len(files)):
            pc = np.loadtxt(files[i])
            patches[i*num_patches_per_pc:(i+1)*num_patches_per_pc, ...] \
                = get_single_pc_patches(pc, patch_size=patch_num_points, num_patch=num_patches_per_pc)

    elif patch_type == 'geo':
        for i in range(len(files)):
            patches[i * num_patches_per_pc:(i + 1) * num_patches_per_pc, ...] \
                = get_single_pc_patches_geo_dist(files[i], os.path.join('data', 'geo_dist'),
                                                 patch_size=patch_num_points, num_patch=num_patches_per_pc)
    return patches


class Fetcher(threading.Thread):
    def __init__(self, opts):
        super(Fetcher, self).__init__()
        self.queue = queue.Queue(50)
        self.stopped = False
        self.opts = opts
        self.use_geodesic_data = False
        self.batch_size = self.opts.batch_size
        self.patch_size = self.opts.patch_size

        files = glob(os.path.join(self.opts.train_test_data_dir, '*.xyz'))
        if self.opts.use_geo:
            self.patches = prepare_train_data(files, patch_size=self.opts.patch_size,
                                                                         num_patch=self.opts.num_patch,
                                                                         patch_type='geo')  # 648*256*3
        else:
            self.patches = prepare_train_data(files, patch_size=self.opts.patch_size,
                                                                         num_patch=self.opts.num_patch,
                                                                         patch_type='normal')  # 648*256*3
        self.x_num_points = int(self.patches.shape[1] / self.opts.up_ratio)
        print('patches shape: ', self.patches.shape)
        self.patches, _, _ = pc_util.normalize_point_cloud(self.patches)
        self.radius_data = np.ones(shape=(self.patches.shape[0]))
        print("All train data has been loaded.")

        self.use_random_input = True
        self.use_fps_data = True
        self.train_patch_size = min(1000, self.patches.shape[0])  # !!!

        self.batch_size = self.opts.batch_size
        if not self.use_fps_data:
            self.sample_cnt = self.train_patch_size
        else:
            self.sample_cnt = self.train_patch_size * 4
        self.num_batches = max(self.sample_cnt // self.batch_size, 1)
        print("NUM_BATCH is %s" % (self.num_batches))

    def run(self):
        """
        self.patches: num_patches * num_points * num_channels
        :return: x: num_patches * x_num_points * num_channels
                y: <==> self.patches
        """
        while not self.stopped:
            indices = np.arange(self.patches.shape[0])
            np.random.shuffle(indices)
            self.patches[:] = self.patches[indices, ...]
            self.radius_data[:] = self.radius_data[indices, ...]

            patches = self.patches[:self.train_patch_size]
            radius_data = self.radius_data[:self.train_patch_size]

            radius = radius_data.copy()
            x = None
            y = None

            self.use_fps_data = True
            self.use_random_input = False
            if self.use_fps_data:
                y_ = patches.copy()  # patches * 256 * 3
                for i in range(y_.shape[0]):
                    if self.use_random_input:
                        for k in range(4):
                            x_r, _ = downsample_points(y_[i], 64)
                            if x is None:
                                x = [x_r]
                            else:
                                x = np.concatenate((x, [x_r]), axis=0)
                            if y is None:
                                y = y_[i:i + 1]
                            else:
                                y = np.concatenate((y, y_[i:i + 1]), axis=0)
                    x_, x_geodesic_ = decompose_points(y_[i:i+1], num_parts=4)  # 4*64*3
                    for k in range(4):           #x_.shape[0]
                        # x_, _ = downsample_points(y_[i], 64)
                        if x is None:
                            x = x_[k:k+1]
                        else:
                            x = np.concatenate((x, x_[k:k+1]), axis=0) #[k:k+1]
                        if y is None:
                            y = y_[i:i + 1]
                        else:
                            y = np.concatenate((y, y_[i:i + 1]), axis=0)

                # radius = np.concatenate((radius, np.ones(shape=(y_.shape[0] * 4))), axis=0)
                radius = np.ones(shape=(y_.shape[0] * 8))
            for batch_idx in range(self.num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                batch_x = x[start_idx:end_idx, :, :].copy()
                batch_y = y[start_idx:end_idx, :, :].copy()
                batch_radius = radius[start_idx:end_idx].copy()
                if batch_x.shape[0] == 0:
                    continue

                if True:
                    batch_x = point_operation.jitter_perturbation_point_cloud(batch_x, sigma=self.opts.jitter_sigma, clip=self.opts.jitter_max)
                    # batch_x, batch_y = point_operation.rotate_point_cloud_and_gt(batch_x, batch_y)

                self.queue.put((batch_x[:, :, :3], batch_y[:, :, :3], batch_radius))

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        print("Shutdown .....")
        while not self.queue.empty():
            self.queue.get()
        print("Remove all queue data")


if __name__ == '__main__':
    pass
