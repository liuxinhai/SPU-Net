#  prepare for the experiment
#  input: *.xyz: (8192, 3)
#  output: *.xyz: (2048, 3)

import os
import numpy as np
from glob import glob

from Common.pc_util import downsample_points_fps, get_knn_idx

import argparse


def init_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./data/test')
    parser.add_argument('--num_parts', type=int, default=4)
    parsers = parser.parse_args()
    return parsers


def decompose_points(points, num_parts=4):
    """
    :param points: num_patches * num_points * num_channels
    :return: num_patches * num_parts * (num_points/num_parts) * num_channels
    """
    part_num_points = int(points.shape[1] / num_parts)
    points = points.astype(np.float32)

    # patches_part_pcs = []
    for j in range(points.shape[0]):
        single_pc = np.expand_dims(points[j, ...], axis=0)
        part_pcs = np.zeros((num_parts, part_num_points, points.shape[2]))
        part_pcs_list = np.zeros((num_parts, part_num_points), dtype=np.int32)
        for i in range(num_parts):
            part_pcs[i, ...], part_pcs_list[i, ...] = downsample_points_fps(single_pc[0], part_num_points)

            idx = np.squeeze(get_knn_idx(part_pcs[i, ...], single_pc[0], 1), axis=1).tolist()
            new_points = np.zeros((single_pc.shape[1] - len(idx), single_pc.shape[2])).astype(np.float32)
            new_points_id = 0
            for id in range(single_pc.shape[1]):
                if id not in idx:
                    new_points[new_points_id, :] = single_pc[0, id, :]
                    new_points_id += 1
            single_pc = np.expand_dims(new_points, axis=0)
        # patches_part_pcs.append(part_pcs)

    return part_pcs, part_pcs_list


if __name__ == '__main__':
    parsers = init_configs()
    num_parts = parsers.num_parts
    input_dir = parsers.input_dir
    files = glob(os.path.join(input_dir, '*.xyz'))
    output_base_dir = os.path.join(input_dir, 'decompose')
    if not os.path.exists(output_base_dir):
        os.mkdir(output_base_dir)

    # open session
    # for file in files:
    file = files[0]
    points = np.loadtxt(file).astype(np.float32)
    points = np.expand_dims(points, axis=0)
    part_num_points = int(points.shape[1] / num_parts)
    output_dir = os.path.join(output_base_dir, os.path.basename(file).split('.')[0])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    part_pcs = decompose_points(points, num_parts)
    np.savetxt(os.path.join(output_dir, 'all.xyz'), points[0], fmt='%.6f')
    for i in range(part_pcs.shape[1]):
        np.savetxt(os.path.join(output_dir, '%d.xyz' % (i)), part_pcs[0, i, ...], fmt='%.6f')
