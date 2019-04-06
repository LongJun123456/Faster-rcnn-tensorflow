# calculation the loss of rpn
# prediction_bbox: rpn_pred_box shape:[-1, 4]
# anchor: generated anchors shape:[-1,4]
# ground_truth: ground_truth boxes shape:[-1, 4]
# probability: rpn_pred_probability shape:[-1, 2]
# lanel: anchor label shape:[-1]  -1 for unuseful label, 1 for positive label, 0 for negative label
# label_gt_order：the groundtruth index tag of anchors, the max IOU with the selected gt



# loss function:cross entry loss\smooth l1 loss

import tensorflow as tf
import numpy as np
import config as cfg


class RPN_loss(object):
    def __init__(self, prediction_bbox, anchor, ground_truth, probability, label, label_gt_order):
        self.prediction_bbox = prediction_bbox
        self.ground_truth = ground_truth
    
        self.label_1 = label

        #self.useful_label: the label with content 0 and 1
        self.useful_label = tf.reshape(tf.where(tf.not_equal(self.label_1, -1)), [-1]) 
        self.reg_loss_nor = tf.cast(tf.shape(self.label_1)[0]/9, tf.float32)    
        
        #gather the label to be computed in cls_loss
        self.label_gather = tf.gather(self.label_1, self.useful_label)
        self.label_gather = tf.cast(self.label_gather, dtype=tf.int32)
        self.label_gt_order = tf.gather(label_gt_order, self.useful_label)  # 每个anchor对应的groundtruth编号，一维
        
        #gather the anchor to be computed  in reg_loss
        self.anchor = tf.gather(anchor, self.useful_label)

        #gather the rpn_probability to be computed in cls_loss
        self.probability = probability
        self.probability = tf.squeeze(self.probability)
        self.probability = tf.reshape(self.probability, [-1,9*2])
        self.probability = tf.reshape(self.probability, [-1,2])
        self.probability_gather = tf.gather(self.probability, self.useful_label)
        self.probability_gather = tf.cast(self.probability_gather, dtype=tf.float32)

         #gather the prediction_bbox to be computed in reg_loss
        self.prediction_bbox = tf.squeeze(self.prediction_bbox)
        self.prediction_bbox = tf.reshape(self.prediction_bbox, [-1,9*4])
        self.prediction_bbox = tf.reshape(self.prediction_bbox, [-1,4])
        self.prediction_bbox_gather = tf.gather(self.prediction_bbox, self.useful_label)

    def reconsitution_coords(self):
        #reconsitution_coords of anchor , ground_truth_box output format[x1,y1,x2,y2]
        self.re_prediction_bbox = self.prediction_bbox_gather
        anchor_x1 = self.anchor[:, 0]
        anchor_y1 = self.anchor[:, 1]
        anchor_x2 = self.anchor[:, 2]
        anchor_y2 = self.anchor[:, 3]

        self.re_anchor_0 = tf.cast((anchor_x2+anchor_x1)/2.0, dtype=tf.float32)
        self.re_anchor_1 = tf.cast((anchor_y2+anchor_y1)/2.0, dtype=tf.float32)
        self.re_anchor_2 = tf.cast((anchor_x2-anchor_x1), dtype=tf.float32)
        self.re_anchor_3 = tf.cast((anchor_y2-anchor_y1), dtype=tf.float32)
        self.re_anchor = tf.squeeze(tf.stack(
            [self.re_anchor_0, self.re_anchor_1, self.re_anchor_2, self.re_anchor_3], axis=1))


        ground_truth_x1 = self.ground_truth[:, 0]
        ground_truth_y1 = self.ground_truth[:, 1]
        ground_truth_x2 = self.ground_truth[:, 2]
        ground_truth_y2 = self.ground_truth[:, 3]

        re_ground_truth_0 = tf.expand_dims(tf.cast((ground_truth_x1+ground_truth_x2)/2.0, dtype=tf.float32),-1)
        re_ground_truth_1 = tf.expand_dims(tf.cast((ground_truth_y1+ground_truth_y2)/2.0, dtype=tf.float32),-1)
        re_ground_truth_2 = tf.expand_dims(tf.cast((ground_truth_x2-ground_truth_x1+1.0), dtype=tf.float32),-1)
        re_ground_truth_3 = tf.expand_dims(tf.cast((ground_truth_y2-ground_truth_y1+1.0), dtype=tf.float32),-1)
        self.re_ground_truth = tf.concat([re_ground_truth_0, re_ground_truth_1, re_ground_truth_2, re_ground_truth_3], axis=1)

        
        #self.gt_map=tf.one_hot(self.label_gt_order,self.size)
        #self.re_label_gt_order=tf.matmul(self.gt_map,self.re_ground_truth)
        #self.re_label_gt_order=tf.cast(self.re_label_gt_order,dtype=tf.float32)
        self.re_label_gt_order = tf.gather(self.re_ground_truth, self.label_gt_order)

        #chosse which rpn_box to be computed in reg_loss, for label is positive ie, 1
        self.label_weight_c = tf.cast((self.label_gather>0), tf.float32)
        self.label_weight_c = tf.expand_dims(self.label_weight_c, axis=1)

    def class_loss(self, p_pred, label):
        #cross entry for cls loss
        l_loss_sum = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_pred, labels=label))
        return l_loss_sum

    def smooth_l1_loss(self, bbox_predicted, bbox_ground_truth,  weight, lmd=1.0, sigma=3.0, dim2mean=1):
        # if the 4 figures of bbox have been calculated
        # weight:to delete negative anchors
        sigma_1 = sigma ** 2
        bbox_ground_truth_0 = tf.cast((bbox_ground_truth[:, 0]-self.re_anchor_0)/self.re_anchor_2, dtype=tf.float32)
        bbox_ground_truth_1 = tf.cast((bbox_ground_truth[:, 1]-self.re_anchor_1)/self.re_anchor_3, dtype=tf.float32)
        bbox_ground_truth_2 = tf.cast(tf.log(bbox_ground_truth[:, 2]/self.re_anchor_2), dtype=tf.float32)
        bbox_ground_truth_3 = tf.cast(tf.log(bbox_ground_truth[:, 3]/self.re_anchor_3), dtype=tf.float32)
        re_bbox_ground_truth = tf.stack([bbox_ground_truth_0, bbox_ground_truth_1, bbox_ground_truth_2, bbox_ground_truth_3], axis=1)
        re_bbox_predicted = bbox_predicted
        bbox_diff = re_bbox_predicted - re_bbox_ground_truth       
        t_diff = bbox_diff*weight
        t_diff_abs = tf.abs(t_diff)
        compare_1 = tf.stop_gradient(tf.to_float(tf.less(t_diff_abs, 1.0/sigma_1)))
        #compare_1 = tf.to_float(tf.less(t_diff_abs, 1.0/sigma_1))
        sl_loss_box = (sigma_1/2.0)*compare_1*tf.pow(t_diff_abs, 2) + (1.0-compare_1)*(t_diff_abs-0.5/sigma_1)
        sum_loss_box = tf.reduce_sum(sl_loss_box)
        loss_box = sum_loss_box*lmd/256	
        print('reg')
        print(sum_loss_box)
        return loss_box

    def add_loss(self):
        self.reconsitution_coords()
        self.log_loss = self.class_loss(self.probability_gather, self.label_gather)
        self.reg_loss = self.smooth_l1_loss(self.re_prediction_bbox, self.re_label_gt_order, self.label_weight_c)
        self.rpn_loss = self.log_loss+self.reg_loss
        tf.summary.scalar('rpn/rpn_log_loss', self.log_loss)
        tf.summary.scalar('rpn/rpn_reg_loss', self.reg_loss)
        return self.rpn_loss
