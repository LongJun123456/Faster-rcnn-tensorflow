# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 18:12:44 2018

@author: LongJun
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
import os
import config as cfg
import tensorflow as tf
from anchor_label import anchor_labels_process
from rois_target import proposal_target
from anchor_generate import all_anchor_conner
import cv2
import copy
slim = tf.contrib.slim
import numpy as np
import tfplot as tfp

"""build the whole network"""
class Net(object):
    def __init__(self, is_training=True, keep_prob=0.5):
        self.num_anchor = len(cfg.anchor_scales) * len(cfg.anchor_ratios)
        self.image = tf.placeholder(tf.float32, shape = [1, None, None, 3])
        self.image_width = tf.placeholder(tf.int64)
        self.image_height = tf.placeholder(tf.int64)
        self.gt_boxes = tf.placeholder(tf.float32, shape = [None, 4])
        self.gt_cls = tf.placeholder(tf.float32, shape=[None])
        self.is_training = is_training
        self.anchor_batch =cfg.anchor_batch
        self.num_classes = len(cfg.CLASSES)
        self.feat_stride = cfg.feat_stride
        self._proposal_targets = {}
        self._predictions = {}
        
        if self.is_training:
            self.weight_file_path = cfg.weight_path
        else:
            self.weight_file_path = cfg.weigt_output_path 
        self.build_network()
        #rpn_loss.__init__(self, self.rois_output['rois_reg'], self.all_anchors, self.gt_boxes, \
        #                  self.rois_output['rois_cls'], self.labels, self.anchor_obj, tf.shape(self.gt_boxes)[0])
        self.add_image_summary()
        
    def proposal_target_layer(self, rois, roi_scores, name):
        """compute the target used for loss convergence"""
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                    proposal_target,
                    [rois, roi_scores, self.gt_boxes, self.num_classes, self.gt_cls],
                    [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                    name="proposal_target")
        rois.set_shape([cfg.dect_train_batch, 5]) #[128,5]
        roi_scores.set_shape([cfg.dect_train_batch])
        labels.set_shape([cfg.dect_train_batch, 1])
        bbox_targets.set_shape([cfg.dect_train_batch, self.num_classes * 4])
        bbox_inside_weights.set_shape([cfg.dect_train_batch, self.num_classes * 4])
        bbox_outside_weights.set_shape([cfg.dect_train_batch, self.num_classes * 4])
        self._proposal_targets['rois'] = rois 
        self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
        self._proposal_targets['bbox_targets'] = bbox_targets
        self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
        self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
        
        return rois, roi_scores
    
    
    def build_network(self):
        self.net_feature = self.vgg16(self.image) #get the vggl6 feature
        self.rois_output = self.rpn_net(self.net_feature, self.num_anchor) #rpn net
        self.all_anchors = tf.py_func(all_anchor_conner, [self.image_width, self.image_height, cfg.feat_stride], tf.float32) #generate anchors
        self.labels, self.anchor_obj = self._anchor_labels_process(self.gt_boxes, self.all_anchors, self.anchor_batch,\
                                                  cfg.overlaps_max, cfg.overlaps_min, self.image_width, self.image_height) #anchor process and generate anchor lable[0,-1,1]
        self.rois_score = self.rois_output['rois_cls']
        self.rois_score = tf.squeeze(self.rois_score)
        self.rois_score = tf.reshape(self.rois_score, [-1,9*2])
        self.rois_score = tf.reshape(self.rois_score, [-1,2])
        self.rois_positive_score = tf.nn.softmax(self.rois_score)
        self.rpn_bbx = self.rois_output['rois_bbx']
        self.rpn_bbx = tf.squeeze(self.rpn_bbx)
        self.rpn_bbx = tf.reshape(self.rpn_bbx, [-1, self.num_anchor*4])
        self.rpn_bbx = tf.reshape(self.rpn_bbx, [-1, 4])
        if self.is_training:
            with tf.variable_scope('roi_process') :
                self.rois_coord, rois_score_process = self.rois_process(self.rois_positive_score, self.rpn_bbx,\
                                                    self.image_width, self.image_height, self.all_anchors)  #roi process
                with tf.control_dependencies([self.labels]):
                    self.rois_coord, _ = self.proposal_target_layer(self.rois_coord, rois_score_process, "rpn_rois")
        else:
            with tf.variable_scope('roi_process') :
                self.rois_coord, rois_score_process = self.rois_process(self.rois_positive_score, self.rpn_bbx,\
                                                    self.image_width, self.image_height, self.all_anchors)
        pool5 = self._crop_pool_layer(self.net_feature, self.rois_coord, "pool5")  #roi pooling layer
        fc7 = self._head_to_tail(pool5, self.is_training)
        with tf.variable_scope('vgg_16'):
            self.cls_prob, self.bbox_pred = self._region_classification(fc7, self.is_training)
        
        
    def vgg16(self, input_image):
        with tf.variable_scope('vgg_16') :
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.0005)):
                net = slim.repeat(input_image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        return net
            
            
    def _head_to_tail(self, pool5, is_training, reuse=None): #dection fc head
        with tf.variable_scope('vgg_16'):
            pool5_flat = slim.flatten(pool5, scope='flatten') 
            fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6') 
            if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, 
                                   scope='dropout6') 
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
            if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, 
                            scope='dropout7')
        return fc7                              
            
    
    def _region_classification(self, fc7, is_training): #dection reg and classify head
        with tf.variable_scope('region_deciton') :
            cls_score = slim.fully_connected(fc7, self.num_classes, 
                                            weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                            trainable=is_training,
                                            weights_regularizer=slim.l2_regularizer(0.0005),
                                            activation_fn=None, scope='cls_score') 
            
            cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
            cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred") 
            bbox_pred = slim.fully_connected(fc7, self.num_classes * 4, 
                                            weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
                                            trainable=is_training,
                                            weights_regularizer=slim.l2_regularizer(0.0005),
                                            activation_fn=None, scope='bbox_pred')  
            self._predictions["cls_score"] = cls_score 
            self._predictions["cls_pred"] = cls_pred  
            self._predictions["cls_prob"] = cls_prob  
            self._predictions["bbox_pred"] = bbox_pred 
            return cls_prob, bbox_pred


    def rpn_net(self, input_feature_map,num_anchor): #rpn net built
        with tf.variable_scope('rpn') :
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                 activation_fn=tf.nn.relu,\
                                 weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01) ,\
                                 weights_regularizer=slim.l2_regularizer(0.0005)):
                 rpn_feature = slim.conv2d(input_feature_map, 512, [3,3], scope='conv6')
                 if self.is_training:
                     self.add_heatmap(rpn_feature, name='rpn_feature')
                 rois_cls = slim.conv2d(rpn_feature, 2*num_anchor, [1,1], padding='VALID', activation_fn=None, scope='conv7')
                 rois_reg = slim.conv2d(rpn_feature, 4*num_anchor, [1,1], padding='VALID', activation_fn=None, scope='conv8')
        return {'rois_cls':rois_cls, 'rois_bbx':rois_reg}

    
        
        
    def _anchor_labels_process(self, boxes, conners, anchor_batch, overlaps_max, overlaps_min,\
                               image_width, image_height):
        return tf.py_func(anchor_labels_process, [boxes, conners, anchor_batch, overlaps_max, overlaps_min,\
                                                  image_width, image_height], [tf.float32, tf.int64])
        
    
    def rois_process(self, rpn_score, rpn_bbox, img_width, img_height, anchors):
        """ roi_process nms, clip"""
        if self.is_training:
            pre_nms_topN = cfg.max_rpn_input_num 
            post_nms_topN = cfg.max_rpn_nms_num
        else:
            pre_nms_topN = cfg.test_max_rpn_input_num 
            post_nms_topN = cfg.test_max_rpn_nms_num  
        nms_thresh = cfg.nms_thresh
        scores = rpn_score[:,1]
        rpn_bbox_pred = rpn_bbox
        proposals = self.coord_transform(anchors, rpn_bbox_pred)
        proposals = self.clip_boxes(proposals, img_width, img_height)
        inds = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)
        boxes = tf.gather(proposals, inds) 
        boxes = tf.to_float(boxes)
        scores_gather = tf.gather(scores, inds)
        scores_gather = tf.reshape(scores, shape=(-1, 1))
        batch_inds = tf.zeros((tf.shape(inds)[0], 1), dtype=tf.float32)
        blob = tf.concat([batch_inds, boxes], 1)
        return blob, scores_gather 
    
    
    def coord_transform (self, anchors, boxes):
#        anchor_x = tf.cast((anchors[:,2] + anchors[:,0])*0.5, tf.float32)
#        anchor_y = tf.cast((anchors[:,3] + anchors[:,1])*0.5, tf.float32)
#        acnhor_w = tf.cast(anchors[:,2] - anchors[:,0]+1.0, tf.float32)
#        acnhor_h = tf.cast(anchors[:,3] - anchors[:,1]+1.0, tf.float32)
        anchors = tf.cast(anchors, tf.float32)
        anchor_x = tf.add(anchors[:,2], anchors[:,0]) * 0.5
        anchor_y = tf.add(anchors[:,3], anchors[:,1]) * 0.5
        acnhor_w = tf.subtract(anchors[:,2], anchors[:,0])+1.0
        acnhor_h = tf.subtract(anchors[:,3], anchors[:,1])+1.0
        boxes = tf.squeeze(boxes)
        boxes = tf.reshape(boxes, [-1,4])
        boxes_x = boxes[:,0]*acnhor_w + anchor_x
        boxes_y = boxes[:,1]*acnhor_h + anchor_y
        boxes_w = tf.exp(boxes[:,2])*acnhor_w
        boxes_h = tf.exp(boxes[:,3])*acnhor_h
        coord_x1 = boxes_x - boxes_w*0.5
        coord_y1 = boxes_y - boxes_h*0.5
        coord_x2 = boxes_x + boxes_w*0.5
        coord_y2 = boxes_y + boxes_h*0.5
        coord_result = tf.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=1)
        return coord_result
    
    
    def clip_boxes(self, boxes, img_width, img_height):
          img_width = tf.cast(img_width, tf.float32)
          img_height = tf.cast(img_height, tf.float32)
          b0 = tf.maximum(tf.minimum(boxes[:, 0], img_width - 1), 0.0) 
          b1 = tf.maximum(tf.minimum(boxes[:, 1], img_height - 1), 0.0)
          b2 = tf.maximum(tf.minimum(boxes[:, 2], img_width - 1), 0.0)
          b3 = tf.maximum(tf.minimum(boxes[:, 3], img_height - 1), 0.0)
          return tf.stack([b0, b1, b2, b3], axis=1) 
          
    def _crop_pool_layer(self, bottom, rois, name): 
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1]) 
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.feat_stride) 
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.feat_stride)
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1)) 
            pre_pool_size = cfg.POOLING_SIZE * 2 #7*2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')    
    
    
    def add_image_summary (self):
        """add roi box that has score over 0.5 into tensorboad"""
        rois_score_pro = self.rois_positive_score[:,1]
        rois = self.rois_output['rois_bbx']
        coord_result = self.coord_transform(self.all_anchors, rois)
        roi_inside = tf.py_func(self.roi_serch, [coord_result, self.image_width, self.image_height], tf.int64)
        coord_result_gather = tf.gather(coord_result, roi_inside)
        rois_score_pro = tf.gather(rois_score_pro, roi_inside)

        inds = tf.image.non_max_suppression(coord_result_gather,rois_score_pro , max_output_size=30, iou_threshold=0.5)
        rois_gather = tf.gather(coord_result_gather, inds)
        rois_score_gather = tf.gather(rois_score_pro, inds)

        img = tf.py_func(self.draw_result, [self.image, rois_gather, rois_score_gather], tf.float32)
        tf.summary.image('rois_socre_over_0.5', img)


    def draw_result(self, im, result, rois_score):
        img = copy.deepcopy(im)
        img[0] = img[0] + cfg.PIXEL_MEANS
        for i in range(len(result)): 
            x1 = result[i][0].astype(np.int32)
            y1 = result[i][1].astype(np.int32)
            x2 = result[i][2].astype(np.int32)
            y2 = result[i][3].astype(np.int32)
            if rois_score[i]>0.5:
                cv2.rectangle(img[0], (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img.astype(np.float32)


    def roi_serch (self, target_boxes, im_width, im_height):
        targets_inside = np.where((target_boxes[:,0]>0)&\
                              (target_boxes[:,2]<im_width)&\
                              (target_boxes[:,1]>0)&\
                              (target_boxes[:,3]<im_height))[0]
        return targets_inside
    
    def add_heatmap(self, feature_maps, name):
        """ add feature maps heatmap into tensorboad"""
        def figure_attention(activation):
            fig, ax = tfp.subplots()
            im = ax.imshow(activation, cmap='jet')
            fig.colorbar(im)
            return fig

        heatmap = tf.reduce_sum(feature_maps, axis=-1) 
        heatmap = tf.squeeze(heatmap, axis=0) 
        tfp.summary.plot(name, figure_attention, [heatmap])


if __name__ == '__main__':
    with tf.Session() as sess:
        net = Net()
        print (net.num_class)
#        model_path = os.path.join('model_pretraIned', 'vgg_16.ckpt')
#        reader = pywrap_tensorflow.NewCheckpointReader(model_path)
#        var_to_shape_map = reader.get_variable_to_shape_map() 
#        input_images = tf.placeholder(tf.float32, [None, 600, 600, 3])
#        variables = tf.global_variables()
#        init = tf.global_variables_initializer()
#        sess.run(init)
#        #var_list = get_var_list()
#        #variables_to_restore = slim.get_variables_to_restore(include=var_list)
#        #print (variables_to_restore)
#        #init = tf.global_variables_initializer()
#        #sess.run(init)
#        #saver = tf.train.Saver(var_list=variables_to_restore)
#        #saver.restore(sess, model_path)
#       # print (sess.run('vgg_16/conv4/conv4_2/biases:0'))
##        for key in variables:
##             if key.name.split(':')[0] in var_to_shape_map:
##                 print (key)
#        print (var_to_shape_map)
             
               
