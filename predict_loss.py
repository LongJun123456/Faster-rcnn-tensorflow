# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:54:54 2018

@author: LongJun
"""
import tensorflow  as tf
""" Predict_loss Class: used to compute the reg_loss and class_loss of the final output
    reg_loss:Smooth L1
    class_loss: cross entry"""
class Predict_loss(object):
    def __init__(self, cls_score, labels, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        self.cls_score = cls_score
        self.labels = labels
        self.bbox_pred = bbox_pred
        self.bbox_targets = bbox_targets
        self.bbox_inside_weights = bbox_inside_weights
        self.bbox_outside_weights = bbox_outside_weights
    
    
    def rcnn_class_loss(self, labels, cls_score):
        cls_score = cls_score
        labels = tf.reshape(labels, [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=labels))
        return cross_entropy
    
    
    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box,axis=dim))
        return loss_box
    
    def add_loss(self):
        log_loss = self.rcnn_class_loss(self.labels, self.cls_score)
        reg_loss = self.smooth_l1_loss(self.bbox_pred, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights)
        total_loss = log_loss + reg_loss
        tf.summary.scalar('fast-rcnn/class_loss',log_loss)
        tf.summary.scalar('fast-rcnn/reg_loss', reg_loss)
        return total_loss