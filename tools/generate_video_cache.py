#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import _init_paths
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ProgressBar import ProgressBar
from model.config import cfg
from model.nms_wrapper import nms
from model.test import im_detect
from nets.resnet_v1 import resnetv1
from utils.timer import Timer

CLASSES = ('__background__',  # always index 0
           '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
           '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020',
           '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030',
           '2031', '2032', '2033', '2034', '2035', '2036', '2037', '2038', '2039', '2040',
           '2041', '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049', '2050',
           '2051', '2052', '2053', '2054', '2055',
           '3001', '3002', '3003', '3004', '3005', '3006', '3007', '3008', '3009', '3010',
           '3011', '3012', '3013', '3014', '3015', '3016', '3017', '3018', '3019', '3020',
           '3021', '3022', '3023', '3024', '3025', '3026', '3027', '3028', '3029', '3030',
           '3031', '3032', '3033', '3034', '3035', '3036', '3037', '3038', '3039', '3040',
           '3041', '3042', '3043', '3044', '3045', '3046', '3047', '3048', '3049', '3050',
           '3051', 'undefined')


def demo(sess, net, image_file, output_dir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_file)

    im_name = os.path.basename(image_file).split(".")[-2]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    recs = dict()
    recs['image_name'] = image_file
    object_list = []

    CONF_THRESH = 0.1
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        det_valid_idx = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in det_valid_idx:
            bnd_object = dict()
            bnd_object['id'] = cls
            bnd_object['bndbox'] = dict()
            bnd_object['bndbox']['xmin'] = float(dets[i, 0])
            bnd_object['bndbox']['ymin'] = float(dets[i, 1])
            bnd_object['bndbox']['xmax'] = float(dets[i, 2])
            bnd_object['bndbox']['ymax'] = float(dets[i, 3])
            object_list.append(bnd_object)

    recs['object_num'] = len(object_list)
    recs['objects'] = object_list
    with open(os.path.join(output_dir, '%s.json' % im_name), 'w') as f:
        json.dump(recs, f, indent=4)



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--image-list', dest='image_list', help='image list')
    parser.add_argument('--output-dir', dest='output_dir', help='each image generates a file in the output directory')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    tfmodel = os.path.join('data', 'voc_2007_trainval', 'default',
                           'res101_faster_rcnn_iter_70000.ckpt')

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network

    net = resnetv1(num_layers=101)
    net.create_architecture("TEST", len(CLASSES),
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    with open(args.image_list, 'r') as f:
        image_json = json.load(f)
        im_names = image_json['image_list']
        bar = ProgressBar(total=len(im_names))
        for im_name in im_names:
            bar.move()
            bar.log('Demo for {}'.format(im_name))
            demo(sess, net, im_name, args.output_dir)

    plt.show()
