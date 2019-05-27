# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:22:57 2018

@author: LongJun
"""
import os
import numpy as np

PASCAL_PATH = 'dataset'

CACHE_PATH = 'annotation_cache'

BATCH_SIZE = 1 

CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

train_imdb_name = 'VOC2007_trainval'  # 'VOC2007_trainval' or 'VOC2007_trainval+VOC2012_trainval' 

test_imdb_name = 'VOC2007_test' #'VOC2007_test'

#max and min img size
target_size = 600

max_size = 1000

#FLIPPED = False

#max training step
MAX_ITER = 70000 #90000 for VOC2007+VOC2012, 70000 for VOC2007 

#the step of LEARNING_RATE decay
lr_change_ITER = 50000 #70000 for VOC2007+VOC2012 50000 for VOC2007

LEARNING_RATE = [0.001, 0.0001]

SUMMARY_ITER = 50

SAVE_ITER = 5000

#threshold for anchor label
overlaps_max = 0.7

overlaps_min = 0.3

OUTPUT_DIR = os.path.join('output')

Summary_output = 'summary_output'
#momentum opitimizer config
momentum = 0.9

GPU_ID = '1'

#anchor size and ratios
anchor_scales = [128,256,512]

anchor_ratios = [0.5,1,2]

anchor_batch = 256

weight_path = os.path.join('model_pretrained', 'vgg_16.ckpt')

weigt_output_path = OUTPUT_DIR

test_output_path = 'test_output'

feat_stride = 16

PIXEL_MEANS = np.array([[[122.7717, 115.9465, 102.9801]]])

#roi nms config 
max_rpn_input_num = 12000

max_rpn_nms_num = 2000

test_max_rpn_input_num = 6000

test_max_rpn_nms_num = 300

nms_thresh = 0.7

#batch for dection network
dect_train_batch = 256

dect_fg_rate = 0.25

bbox_nor_target_pre = True

bbox_nor_mean = (0.0, 0.0, 0.0, 0.0)

bbox_nor_stdv = (0.1, 0.1, 0.2, 0.2)

roi_input_inside_weight = (1.0, 1.0, 1.0, 1.0)

POOLING_SIZE = 7

#threshold for roi
fg_thresh = 0.5

bg_thresh_hi = 0.5

bg_thresh_lo = 0.0

test_nms_thresh = 0.3

test_fp_tp_thresh = 0.5

test_max_per_image = 100

#test_image_show num
img_save_num = 2

image_output_dir = os.path.join(test_output_path, 'image_output')
