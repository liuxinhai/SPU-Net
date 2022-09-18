# -*- coding: utf-8 -*-
# xinhai liu

import tensorflow as tf
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
from tf_ops.nn_distance import tf_nndistance
from tf_ops.approxmatch import tf_approxmatch

from Common import pc_util
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point, knn_point_2
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
import numpy as np
import math


def knn_extractor(adj_matrix, k=5):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int
    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    dis, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx, dis


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def pairwise_distance_gt(point_cloud, point_cloud_gt):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud_gt, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True), perm=[0, 2, 1])

    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def get_neighbors(point_cloud, nn_idx):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int
    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)

    return point_cloud_neighbors


def get_plane_constraint_loss(point_cloud, k=5):
    batch_size = point_cloud.get_shape().as_list()[0]
    adj_matrix = pairwise_distance(point_cloud)
    nn_idx, _ = knn_extractor(adj_matrix, k=k)
    point_neighbors = get_neighbors(point_cloud, nn_idx)
    mean_ponit = tf.reduce_mean(point_neighbors, axis=2, keep_dims=True)
    mean_ponits = tf.tile(mean_ponit, [1, 1, k, 1])
    point_cloud_transpose = tf.transpose(point_neighbors, perm=[0, 1, 3, 2])
    point_cloud_inner = tf.matmul(mean_ponits, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(mean_ponits), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(tf.reduce_sum(tf.square(point_neighbors), axis=-1, keep_dims=True),
                                               [0, 1, 3, 2])
    loss = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    loss = tf.reduce_sum(tf.reshape(loss, [-1]), axis=0)
    return loss / (k * k)

def get_plane_constraint_loss_minimize(point_cloud, k=5):
    batch_size = point_cloud.get_shape().as_list()[0]
    adj_matrix = pairwise_distance(point_cloud)
    nn_idx, _ = knn_extractor(adj_matrix, k=k)
    point_neighbors = get_neighbors(point_cloud, nn_idx)
    mean_ponit = tf.reduce_mean(point_neighbors, axis=2, keep_dims=True)
    mean_ponits = tf.tile(mean_ponit, [1, 1, k, 1])

    point_cloud_transpose = tf.transpose(point_neighbors, perm=[0, 1, 3, 2])
    point_cloud_inner = tf.matmul(mean_ponits, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(mean_ponits), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(tf.reduce_sum(tf.square(point_neighbors), axis=-1, keep_dims=True),
                                               [0, 1, 3, 2])
    loss_target = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    pc_raw = tf.tile(tf.expand_dims(point_cloud, axis=2), [1, 1, k, 1])
    pc_raw_inner = tf.matmul(pc_raw, point_cloud_transpose)
    pc_raw_inner = -2 * pc_raw_inner
    poc_raw_square = tf.reduce_sum(tf.square(pc_raw), axis=-1, keep_dims=True)
    loss_true = poc_raw_square + pc_raw_inner + point_cloud_square_tranpose

    loss = tf.reduce_sum(tf.reshape(tf.abs(loss_target-loss_true), [-1]), axis=0)
    return loss / (k * k)

def get_projection_loss(point_cloud, k=5):
    batch_size = point_cloud.get_shape().as_list()[0]
    num_points = point_cloud.get_shape().as_list()[1]
    adj_matrix = pairwise_distance(point_cloud)
    nn_idx, _ = knn_extractor(adj_matrix, k=k)
    point_neighbors = get_neighbors(point_cloud, nn_idx)
    mean_ponit = tf.reduce_mean(point_neighbors, axis=2, keep_dims=True)
    mean_ponits = tf.tile(mean_ponit, [1, 1, k, 1])

    point_cloud_transpose = tf.transpose(point_neighbors, perm=[0, 1, 3, 2])
    point_cloud_inner = tf.matmul(mean_ponits, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(mean_ponits), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(tf.reduce_sum(tf.square(point_neighbors), axis=-1, keep_dims=True),
                                               [0, 1, 3, 2])
    loss_target = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    pc_raw = tf.tile(tf.expand_dims(point_cloud, axis=2), [1, 1, k, 1])
    pc_raw_inner = tf.matmul(pc_raw, point_cloud_transpose)
    pc_raw_inner = -2 * pc_raw_inner
    poc_raw_square = tf.reduce_sum(tf.square(pc_raw), axis=-1, keep_dims=True)
    loss_true = poc_raw_square + pc_raw_inner + point_cloud_square_tranpose

    final_loss = tf.abs(loss_target - loss_true)
    weights = tf.reduce_sum(loss_true, axis=-2, keepdims=True)
    weights = tf.square(weights)
    constant = tf.ones([batch_size, num_points, 1, k], dtype=tf.float32)
    weights = weights + constant
    weights = tf.reciprocal(weights)
    final_loss = tf.reduce_sum(final_loss, axis=-2, keepdims=True)
    final_loss = tf.multiply(weights, final_loss)

    loss = tf.reduce_sum(tf.reshape(final_loss, [-1]), axis=0)
    return loss / (k * k)

def get_plane_constraint_loss_minimize_gt(point_cloud, point_cloud_gt, k=5):
    batch_size = point_cloud.get_shape().as_list()[0]
    adj_matrix = pairwise_distance_gt(point_cloud, point_cloud_gt)
    nn_idx, _ = knn_extractor(adj_matrix, k=k)
    point_neighbors = get_neighbors(point_cloud_gt, nn_idx)
    mean_ponit = tf.reduce_mean(point_neighbors, axis=2, keep_dims=True)
    mean_ponits = tf.tile(mean_ponit, [1, 1, k, 1])

    point_cloud_transpose = tf.transpose(point_neighbors, perm=[0, 1, 3, 2])
    point_cloud_inner = tf.matmul(mean_ponits, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(mean_ponits), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(tf.reduce_sum(tf.square(point_neighbors), axis=-1, keep_dims=True),
                                               [0, 1, 3, 2])
    loss_target = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    pc_raw = tf.tile(tf.expand_dims(point_cloud, axis=2), [1, 1, k, 1])
    pc_raw_inner = tf.matmul(pc_raw, point_cloud_transpose)
    pc_raw_inner = -2 * pc_raw_inner
    poc_raw_square = tf.reduce_sum(tf.square(pc_raw), axis=-1, keep_dims=True)
    loss_true = poc_raw_square + pc_raw_inner + point_cloud_square_tranpose

    loss = tf.reduce_sum(tf.reshape(tf.abs(loss_target-loss_true), [-1]), axis=0)
    return loss / (k * k)

def pc_distance(pcd1, pcd2, dis_type='emd', radius=1):
    if dis_type == 'cd':
        return chamfer(pcd1, pcd2, radius=radius)
    else:
        return earth_mover(pcd1, pcd2, radius=radius)


def classify_loss(pre_label, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_label, labels=label)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss


def chamfer(pcd1, pcd2, radius=1.0):
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pcd1, pcd2)

    """
    CD_dist_forward = 0.5 * dists_forward
    CD_dist_forward = tf.reduce_mean(CD_dist_forward, axis=1)
    CD_dist_forward = CD_dist_forward / radius
    CD_dist_forward = tf.reduce_mean(CD_dist_forward)
    CD_dist_backward = 0.5 * dists_backward
    CD_dist_backward = tf.reduce_mean(CD_dist_backward, axis=1)
    CD_dist_backward = CD_dist_backward / radius
    CD_dist_backward = tf.reduce_mean(CD_dist_backward)
    cd_loss = CD_dist_forward + CD_dist_backward
    """
    # CD_dist = 0.5 * tf.reduce_mean(dists_forward, axis=1) + 0.5 * tf.reduce_mean(dists_backward, axis=1)
    CD_dist = 0.5 * dists_forward + 0.5 * dists_backward
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist / radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss


def earth_mover(pcd1, pcd2, radius=1.0):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    cost = cost / radius
    return tf.reduce_mean(cost / num_points)


def py_uniform_loss(points, idx, pts_cn, radius):
    # print(type(idx))
    B, N, C = points.shape
    _, npoint, nsample = idx.shape
    uniform_vals = []
    for i in range(B):
        point = points[i]
        for j in range(npoint):
            number = pts_cn[i, j]
            coverage = np.square(number - nsample) / nsample
            if number < 5:
                uniform_vals.append(coverage)
                continue
            _idx = idx[i, j, :number]
            disk_point = point[_idx]
            if disk_point.shape[0] < 0:
                pair_dis = pc_util.get_pairwise_distance(disk_point)  # (batch_size, num_points, num_points)
                nan_valid = np.where(pair_dis < 1e-7)
                pair_dis[nan_valid] = 0
                pair_dis = np.squeeze(pair_dis, axis=0)
                pair_dis = np.sort(pair_dis, axis=1)
                shortest_dis = np.sqrt(pair_dis[:, 1])
            else:
                shortest_dis = pc_util.get_knn_dis(disk_point, disk_point, 2)
                shortest_dis = shortest_dis[:, 1]
            disk_area = math.pi * (radius ** 2) / disk_point.shape[0]
            # expect_d = math.sqrt(disk_area)
            expect_d = np.sqrt(2 * disk_area / 1.732)  # using hexagon
            dis = np.square(shortest_dis - expect_d) / expect_d
            uniform_val = coverage * np.mean(dis)

            uniform_vals.append(uniform_val)

    uniform_dis = np.array(uniform_vals).astype(np.float32)

    uniform_dis = np.mean(uniform_dis)
    return uniform_dis


# whole version, slower
def get_uniform_loss2(pcd, percentages=[0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.015], radius=1.0):
    B, N, C = pcd.get_shape().as_list()
    npoint = int(N * 0.05)
    loss = []
    for p in percentages:
        nsample = int(N * p)
        r = math.sqrt(p * radius)
        # print(npoint,nsample)
        new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
        idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)  # (batch_size, npoint, nsample)

        uniform_val = tf.py_func(py_uniform_loss, [pcd, idx, pts_cnt, r], tf.float32)

        loss.append(uniform_val * math.sqrt(p * 100))
    return tf.add_n(loss) / len(percentages)


# [0.004,0.006,0.008,0.010,0.012]
# [0.006,0.008,0.010,0.012,0.015]
# [0.010,0.012,0.015,0.02,0.025]
# simplfied version, faster
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

def get_uniform_loss(pcd, percentages=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0):
    B, N, C = pcd.get_shape().as_list()
    npoint = int(N * 0.05)
    loss = []
    len_percentages = len(percentages)
    for p in percentages:
        nsample = int(N * p)
        if nsample == 0:
            len_percentages -= 1
            continue
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) * p / nsample
        # print(npoint,nsample)
        new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
        idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)  # (batch_size, npoint, nsample)

        # expect_len =  tf.sqrt(2*disk_area/1.732)#using hexagon
        expect_len = tf.sqrt(disk_area)  # using square

        grouped_pcd = group_point(pcd, idx)
        grouped_pcd = tf.concat(tf.unstack(grouped_pcd, axis=1), axis=0)

        if grouped_pcd.shape[1] < 2:
            len_percentages -= 1
            continue

        var, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]
        uniform_dis = tf.sqrt(tf.abs(uniform_dis + 1e-8))
        uniform_dis = tf.reduce_mean(uniform_dis, axis=[-1])
        uniform_dis = tf.expand_dims(uniform_dis, axis=-1)
        uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
        uniform_dis = tf.reshape(uniform_dis, [-1])


        mean, variance = tf.nn.moments(uniform_dis, axes=0)
        mean = mean * math.pow(p * 100, 2)
        # nothing 4
        loss.append(mean)
    return tf.add_n(loss) / len_percentages


# [0.004,0.006,0.008,0.010,0.012]
# [0.006,0.008,0.010,0.012,0.015]
# [0.010,0.012,0.015,0.02,0.025]
# simplfied version, faster
def get_uniform_loss_diy(pcd, percentages=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0):
    B, N, C = pcd.get_shape().as_list()
    npoint = int(N * 0.05)
    loss = []
    len_percentages = len(percentages)
    for p in percentages:
        nsample = int(N * p)
        if nsample == 0:
            len_percentages -= 1
            continue
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) * p / nsample
        # print(npoint,nsample)
        new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
        idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)  # (batch_size, npoint, nsample)

        # expect_len =  tf.sqrt(2*disk_area/1.732)#using hexagon
        expect_len = tf.sqrt(disk_area)  # using square

        grouped_pcd = group_point(pcd, idx)
        grouped_pcd = tf.concat(tf.unstack(grouped_pcd, axis=1), axis=0)

        if grouped_pcd.shape[1] < 2:
            len_percentages -= 1
            continue

        var, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]
        uniform_dis = tf.sqrt(tf.abs(uniform_dis + 1e-8))
        uniform_dis = tf.reduce_mean(uniform_dis, axis=[-1])
        uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
        uniform_dis = tf.reshape(uniform_dis, [-1])

        mean, variance = tf.nn.moments(uniform_dis, axes=0)
        mean = mean * math.pow(p * 100, 2)
        # nothing 4
        loss.append(mean)
    print('******* ', len_percentages)
    return tf.add_n(loss) / len_percentages


def get_repulsion_loss(pred, nsample=20, radius=0.07, knn=False, use_l1=False, h=0.001):
    if knn:
        _, idx = knn_point_2(nsample, pred, pred)
        pts_cnt = tf.constant(nsample, shape=(30, 1024))
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    # get the uniform loss
    if use_l1:
        dists = tf.reduce_sum(tf.abs(grouped_pred), axis=-1)
    else:
        dists = tf.reduce_sum(grouped_pred ** 2, axis=-1)

    val, idx = tf.nn.top_k(-dists, 5)
    val = val[:, :, 1:]  # remove the first one

    if use_l1:
        h = np.sqrt(h) * 2
    print(("h is ", h))

    val = tf.maximum(0.0, h + val)  # dd/np.sqrt(n)
    repulsion_loss = tf.reduce_mean(val)
    return repulsion_loss


##################################################################################
# Loss function
##################################################################################

def discriminator_loss(D, input_real, input_fake, Ra=False, gan_type='lsgan'):
    real = D(input_real)
    fake = D(input_fake)
    real_loss = tf.reduce_mean(tf.square(real - 1.0))
    fake_loss = tf.reduce_mean(tf.square(fake))

    loss = real_loss + fake_loss

    return loss


def generator_loss(D, input_fake):
    fake = D(input_fake)

    fake_loss = tf.reduce_mean(tf.square(fake - 1.0))
    return fake_loss


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss
