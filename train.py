# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:54:12 2018

@author: LongJun
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config as cfg
import os
import pascal_voc as pascl
import tensorflow.contrib.slim as slim
import anchor_generate
from tensorflow.python import pywrap_tensorflow
from anchor_label import anchor_labels_process, labels_generate, anchor_labels_process
import numpy as np
import tensorflow as tf
import network
import datetime
from losslayer import RPN_loss
from predict_loss import Predict_loss
# Solver Class, used for training
# net: the name of backbone net, only support VGG16, more backbone net will be supported in the feature
# data/val_data： trian_data/vla_data is a list that consist of dict of ground_truth label
# rpn_loss class : used for calculating rpn_loss
# predict_loss: used for calculating predict_loss
class Solver(object):   
    def __init__(self, net ,data, val_data, rpn_loss, predict_loss): 
        self.net = net
        self.data = data
        self.val_data = val_data
        self.max_iter = cfg.MAX_ITER
        self.lr = cfg.LEARNING_RATE
        self.rpn_loss = rpn_loss
        self.predict_loss = predict_loss
        self.lr_change_ITER = cfg.lr_change_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.overlaps_max = cfg.overlaps_max
        self.overlaps_min = cfg.overlaps_min
        self._variables_to_fix = {}
        self.Summary_output = os.path.join(cfg.Summary_output, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os._exists(self.Summary_output):
            os.mkdir(self.Summary_output)
        self.train_summary_dir = os.path.join(self.Summary_output, 'train')
        self.val_summary_dir = os.path.join(self.Summary_output, 'val')
        self.model_output_dir = os.path.join(cfg.OUTPUT_DIR) 
        if not os.path.exists(self.model_output_dir):
            os.mkdir(self.model_output_dir)
        if not os.path.exists(self.train_summary_dir):
            os.mkdir(self.train_summary_dir)
        if not os.path.exists(self.val_summary_dir):
            os.mkdir(self.val_summary_dir)
        self.ckpt_filename = os.path.join(self.model_output_dir, 'output.model')
        
 # training process       
    def train_model(self):
        lr = tf.Variable(self.lr[0],trainable=False)
        self.optimizer = tf.train.MomentumOptimizer(lr, cfg.momentum)
        #self.optimizer = tf.train.GradientDescentOptimizer(lr)
        self.loss = self.rpn_loss.add_loss() + self.predict_loss.add_loss()     
        train_op = self.optimizer.minimize(self.loss)
        variables = tf.global_variables()
        reader = pywrap_tensorflow.NewCheckpointReader(self.net.weight_file_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        variables_to_restore = self.get_var_list(variables, var_to_shape_map)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(var_list=variables_to_restore)
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.train_summary_dir, sess.graph)
            val_writer = tf.summary.FileWriter(self.val_summary_dir)
            sess.run(init)
            saver.restore(sess, self.net.weight_file_path)
            self.fix_variables(sess, self.net.weight_file_path)
            saver = tf.train.Saver(variables,max_to_keep = 10)
            for step in range(self.max_iter+1):
                if step == self.lr_change_ITER:
                    lr = tf.assign(lr, self.lr[1])
                train_data = self.data.get()
                image_height = np.array(train_data['image'].shape[1])
                image_width = np.array(train_data['image'].shape[2])
                feed_dict = {self.net.image: train_data['image'], self.net.image_width: image_width,\
                             self.net.image_height: image_height, self.net.gt_boxes: train_data['box'],\
                             self.net.gt_cls: train_data['cls']}
                if step % self.summary_iter == 0:
                    total_loss, summary, learning_rate= sess.run([self.loss, merged, lr], feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    val_data = self.val_data.get()
                    val_image_height = np.array(val_data['image'].shape[1])
                    val_image_width = np.array(val_data['image'].shape[2])
                    val_feed_dict = {self.net.image: val_data['image'], self.net.image_width: val_image_width,\
                                     self.net.image_height: val_image_height, self.net.gt_boxes: val_data['box'],\
                                     self.net.gt_cls: val_data['cls']}
                    val_loss, val_summary = sess.run([self.loss, merged], feed_dict=val_feed_dict)
                    val_writer.add_summary(val_summary, step)
                    print ('The', step, 'step train_total_loss is', total_loss, 'val_total_loss is', val_loss)
                    print ('learning_rate is ', learning_rate)
                if step % self.save_iter == 0:
                    saver.save(sess, self.ckpt_filename, global_step = step)
                sess.run(train_op, feed_dict=feed_dict)
                    
               
                
                
#get the variables to restore               
    def get_var_list(self, global_variables, ckpt_variables):
        variables_to_restore = []
        for key in global_variables:
            print (key.name)
            if key.name == ('vgg_16/fc6/weights:0') or key.name == ('vgg_16/fc7/weights:0'):
                self._variables_to_fix[key.name] = key
                continue
            
            if key.name.split(':')[0] in ckpt_variables:
                variables_to_restore.append(key) 
        return variables_to_restore
    
#because fc6 and fc7 layers of pretrained vgg16 model is convolution format, so we need convert them to fully-connected layers
    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
            with tf.device("/cpu:0"):
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                restorer_fc = tf.train.Saver({'vgg_16' + "/fc6/weights": fc6_conv, 
                                              'vgg_16' + "/fc7/weights": fc7_conv})
                restorer_fc.restore(sess, pretrained_model)
        
                sess.run(tf.assign(self._variables_to_fix['vgg_16' + '/fc6/weights:0'], tf.reshape(fc6_conv, 
                                    self._variables_to_fix['vgg_16' + '/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16' + '/fc7/weights:0'], tf.reshape(fc7_conv, 
                                    self._variables_to_fix['vgg_16' + '/fc7/weights:0'].get_shape())))
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    net = network.Net()
    rpn_loss_obj = RPN_loss(net.rois_output['rois_bbx'], net.all_anchors, net.gt_boxes, \
                        net.rois_output['rois_cls'], net.labels, net.anchor_obj)
    predict_loss = Predict_loss(net._predictions["cls_score"], net._proposal_targets['labels'],\
                                net._predictions['bbox_pred'], net._proposal_targets['bbox_targets'],\
                                net._proposal_targets['bbox_inside_weights'], net._proposal_targets['bbox_outside_weights'])
    
    train_data = pascl.pascal_voc(cfg.train_imdb_name, 'train', fliped=True)
    val_data = pascl.pascal_voc(cfg.test_imdb_name, 'test', fliped=False)
    solver = Solver(net, train_data, val_data, rpn_loss_obj, predict_loss)
    print ('start training')
    solver.train_model()
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
