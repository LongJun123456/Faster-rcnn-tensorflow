# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:18:37 2018

@author: LongJun
"""

import numpy as np
import tensorflow as tf
import config as cfg

def generate_anchors(anchor_scales=[128,256,512], anchor_ratios=[0.5,1,2], anchor_bias_x_ctr=8, anchor_bias_y_ctr=8):
    anchor_width = np.array(anchor_scales)
    anchor_length = np.array(anchor_scales)
    anchor_ratios = np.array(anchor_ratios)
    bias_x_ctr = anchor_bias_x_ctr
    bias_y_ctr = anchor_bias_y_ctr
    anchor_scales = np.stack((anchor_width, anchor_length), axis=-1)
    anchor_size = ratios_process(anchor_scales, anchor_ratios)
    anchor_conner = generate_anchors_conner(anchor_size, bias_x_ctr, bias_y_ctr)
    return anchor_conner



def ratios_process(anchor_scales, anchor_ratios):
    anchor_area = anchor_scales[:,0] * anchor_scales[:,1]
    anchors = np.vstack([get_anchor_size(anchor_area[i], anchor_ratios) for i in range(anchor_area.shape[0])])
    return anchors
    
def get_anchor_size(anchor_area, anchor_ratios):
    width = np.round(np.sqrt(anchor_area/anchor_ratios))
    length = width * anchor_ratios
    anchors = np.stack((width, length), axis=-1)
    return anchors

def generate_anchors_conner(anchor_size, x_ctr, y_ctr):
    width = anchor_size[:,0]
    length = anchor_size[:,1]
    x1 = np.round(x_ctr - 0.5*width)
    y1 = np.round(y_ctr -0.5*length)
    x2 = np.round(x_ctr + 0.5*width)    
    y2 = np.round(y_ctr +0.5*length)
    conners = np.stack((x1, y1, x2, y2), axis=-1)
    #print (conners)
    return conners


def all_anchor_conner(image_width, image_height, stride=16):
    bias_anchor_conner = generate_anchors(cfg.anchor_scales, cfg.anchor_ratios)
    #print (bias_anchor_conner.shape)
    stride = np.float32(stride)
    #return 0
    #dmap_width = tf.to_int32(tf.ceil(image_width/stride))
    #dmap_height = tf.to_int32(tf.ceil(image_height/stride))
    dmap_width = np.ceil(image_width/stride)
    dmap_height = np.ceil(image_height/stride)
    total_pos = (dmap_height*dmap_width).astype(np.int32)
    #offset_x = tf.range(dmap_width) * stride
    #offset_y = tf.range(dmap_height) * stride
    offset_x = np.arange(dmap_width) * stride
    offset_y = np.arange(dmap_height) * stride
    #x,y = tf.meshgrid(offset_x,offset_y)
    x,y = np.meshgrid(offset_x,offset_y)
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    coordinate = np.stack((x, y, x, y), axis=-1)
    #coordinate = tf.reshape(coordinate, [total_pos,1,4])
    #coordinate = tf.reshape(coordinate, [total_pos,4])
    coordinate= np.transpose(np.reshape(coordinate, [1, total_pos, 4]), (1, 0, 2))
    #print (coordinate)
    all_anchor_conners = coordinate + bias_anchor_conner
    all_anchor_conners = np.reshape(all_anchor_conners, [-1,4])
    return np.array(all_anchor_conners).astype(np.float32)

if __name__ == '__main__':
   # all_anchor_conner()
    a = np.array(600)
    b = np.array(800)
    image_width = tf.placeholder(tf.int32)
    image_height = tf.placeholder(tf.int32)
    with tf.Session() as sess:
        conners = tf.py_func(all_anchor_conner, [image_width,image_height,16], tf.float32)
        feed_dict = {image_width:a, image_height:b}
        conners = sess.run(conners, feed_dict=feed_dict)
    print (conners.shape)
