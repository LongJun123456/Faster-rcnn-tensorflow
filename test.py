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
import numpy as np
import tensorflow as tf
import network
import datetime
import cv2
from nms import py_cpu_nms
#Val_test class is used to test the output model, the result is mean_ap tested on the pascal_voc 2007 test_imdb
#net： VGG16
#val_data: val_data name
class Val_test(object):   
    def __init__(self, net ,val_data):
        self.net = net
        self.val_data = val_data
        self.overlaps_max = cfg.overlaps_max
        self.overlaps_min = cfg.overlaps_min
        self.ckpt_filename = tf.train.latest_checkpoint(os.path.join(cfg.OUTPUT_DIR))
        self.test_output_dir = cfg.test_output_path
        self.image_output_dir = cfg.image_output_dir
        txtname = os.path.join(self.val_data.devkil_path, self.val_data.name, 'ImageSets', 'Main', self.val_data.phase+'.txt')
        with open(txtname) as f:
            self.image_index = [x.strip() for x in f.readlines()]

    def test_model(self):
        saver = tf.train.Saver()
        _rois_coord = self.net.rois_coord[:,1:5]
        #rois_coord = self.net.rois_coord
        _pred_box = self.net.bbox_pred
        _pred_score = self.net.cls_prob
        #_pred_box_score_arg = tf.argmax(_pred_score, axis=1)
        dect_total_result = [[[] for i in range(self.val_data.num_gtlabels)] for j in range(self.net.num_classes)]
        with tf.Session() as sess:
            saver.restore(sess, self.ckpt_filename)
            for i in range (self.val_data.num_gtlabels):
                print (i, ' image test compeleted')            
                train_data = self.val_data.get()
                image_height = np.array(train_data['image'].shape[1])
                image_width = np.array(train_data['image'].shape[2])
                feed_dict = {self.net.image: train_data['image'], self.net.image_width: image_width,\
                             self.net.image_height: image_height}
                                
                rois_coord, pred_box, pred_score= sess.run([_rois_coord, _pred_box, _pred_score],\
                                                                        feed_dict=feed_dict) 

                for k in range(1, self.net.num_classes):    
                    #pre_class_arg = np.where(pred_score[:,k]>=0)[0]
                    cls_pred_box_target = pred_box[:, k*4:(k+1)*4]
                    cls_pred_box_target = cls_pred_box_target * np.array(cfg.bbox_nor_stdv) + np.array(cfg.bbox_nor_mean)
                    cls_pred_box_coord = self.coord_transform_inv(rois_coord, cls_pred_box_target.astype(np.float32))
                    cls_pred_box_coord = cls_pred_box_coord/train_data['scale'] + 1.0
                    cls_pred_score = pred_score[:, k]
                    #print(cls_pred_box_coord.shape, cls_pred_score.shape)
                    cls_pred_score = cls_pred_score[:, np.newaxis]  
                    cls_pred_target = np.concatenate((cls_pred_box_coord, cls_pred_score), axis=1)
                    keep = py_cpu_nms(cls_pred_target, cfg.test_nms_thresh)
                    cls_pred_target = cls_pred_target[keep, :]
                    dect_total_result[k][i] = cls_pred_target
                   # print (cls_pred_target)
                image_scores = np.hstack([dect_total_result[j][i][:, -1] for j in range(1, self.net.num_classes)]) #
                if len(image_scores) > cfg.test_max_per_image:
                    image_thresh = np.sort(image_scores)[-cfg.test_max_per_image] #
                    for j in range(1, self.net.num_classes):
                        keep = np.where(dect_total_result[j][i][:, -1] >= image_thresh)[0]
                        dect_total_result[j][i] = dect_total_result[j][i][keep, :] #
            mean_ap = self.map_compute(dect_total_result)
            print ('the mean_ap of pascal_voc 2007 is', mean_ap)
        
        
    def coord_transform_inv (self, anchors, boxes):
        anchors = anchors.astype(np.float32)
        anchors = np.reshape(anchors, [-1,4])
        anchor_x = (anchors[:,2] + anchors[:,0]) * 0.5
        anchor_y = (anchors[:,3] + anchors[:,1]) * 0.5
        acnhor_w = (anchors[:,2] - anchors[:,0]) + 1.0
        acnhor_h = (anchors[:,3] - anchors[:,1]) + 1.0
        boxes = np.reshape(boxes, [-1,4])
        boxes_x = boxes[:,0]*acnhor_w + anchor_x
        boxes_y = boxes[:,1]*acnhor_h + anchor_y
        boxes_w = np.exp(boxes[:,2])*acnhor_w
        boxes_h = np.exp(boxes[:,3])*acnhor_h
        coord_x1 = boxes_x - boxes_w*0.5
        coord_y1 = boxes_y - boxes_h*0.5
        coord_x2 = boxes_x + boxes_w*0.5
        coord_y2 = boxes_y + boxes_h*0.5
        coord_result = np.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=1)
        return coord_result              


#computing map using pascal_voc 2010 algorithm
    def map_compute(self, dect_boxes):
        ap = []
        for cls_ind, cls in enumerate(self.val_data.classes):
            cls_obj = {}
            num_cls_obj = 0
            if cls == 'background':
                 continue
            if not os.path.exists(self.test_output_dir):
                os.mkdir(self.test_output_dir)
            cls_filename = os.path.join(self.test_output_dir, cls+'.txt')
            with open(cls_filename, 'w') as f:
                 for img_ind_dex, image_ind in enumerate(self.image_index):
                      dect_box = dect_boxes[cls_ind][img_ind_dex]
                      if dect_box == []:
                           continue
                      for i in range(dect_box.shape[0]):
                           f.write('{:s} {:2f} {:2f} {:2f} {:2f} {:3f} \n'.format\
                                   (image_ind, dect_box[i][0], dect_box[i][1], dect_box[i][2],\
                                    dect_box[i][3], dect_box[i][4]))
                           
            for gt_label in self.val_data.gt_labels:
                 gt_label_cls_ind = np.where(gt_label['gt_classs']==cls_ind)[0]
                 gt_label_pick_box = gt_label['boxes'][gt_label_cls_ind, :]
                 gt_label_pick_cls = gt_label['gt_classs'][gt_label_cls_ind]
                 diff_pick = gt_label['diff'][gt_label_cls_ind].astype(np.bool)
                 dec_id = [False] * gt_label_cls_ind.size
                 num_cls_obj = num_cls_obj + sum(~diff_pick)
                 cls_obj[gt_label['image_index']] = {'bbox': gt_label_pick_box,\
                                                      'cls': gt_label_pick_cls,\
                                                      'dec_id': dec_id, 'diff': diff_pick}
                 #print (num_cls_obj)
            with open(cls_filename, 'r') as f:
              lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines] 
            image_ids = [x[0] for x in splitlines]  
            confidence = np.array([float(x[5]) for x in splitlines])
            BB = np.array([[float(z) for z in x[1:5]] for x in splitlines]) #bounding box                 
            
            nd = len(image_ids)                   
            tp = np.zeros(nd)         
            fp = np.zeros(nd)
            
            if BB.shape[0] > 0:
                 sorted_ind = np.argsort(-confidence) 
                 BB = BB[sorted_ind, :]
                 image_ids = [image_ids[x] for x in sorted_ind] 
                 for d in range(nd): 
                      R = cls_obj[image_ids[d]] 
                      bb = BB[d, :].astype(float) 
                      ovmax = -np.inf 
                      BBGT = R['bbox'].astype(float) 
                      
                      if BBGT.size > 0: 
                           ixmin = np.maximum(BBGT[:, 0], bb[0])
                           iymin = np.maximum(BBGT[:, 1], bb[1])
                           ixmax = np.minimum(BBGT[:, 2], bb[2])
                           iymax = np.minimum(BBGT[:, 3], bb[3])
                           iw = np.maximum(ixmax - ixmin + 1., 0.)
                           ih = np.maximum(iymax - iymin + 1., 0.)
                           inters = iw * ih
                           uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +\
                                  (BBGT[:, 2] - BBGT[:, 0] + 1.) *\
                                  (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                           overlaps = inters / uni
                           ovmax = np.max(overlaps) 
                           jmax = np.argmax(overlaps) 
                           
                      if ovmax > cfg.test_fp_tp_thresh:
                           if not R['diff'][jmax]:
                               if not R['dec_id'][jmax]: 
                                    tp[d] = 1.
                                    R['dec_id'][jmax] = 1 
                               else:
                                    fp[d] = 1.
                      else:
                           fp[d] = 1. 
            fp = np.cumsum(fp) 
            tp = np.cumsum(tp) 
            rec = tp / float(num_cls_obj) 
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) 
            ap.append(self.val_data.voc_ap(rec, prec)) 	
            print (np.mean(ap))
        return sum(ap)/(self.net.num_classes - 1.0)
    
                   
    def get_var_list(self, global_variables, ckpt_variables):
        variables_to_restore = []
        for key in global_variables:
            if key.name.split(':')[0] in ckpt_variables:
                variables_to_restore.append(key) 
        return variables_to_restore
    
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    net = network.Net(is_training=False)
    val_data = pascl.pascal_voc(cfg.test_imdb_name, 'test', fliped=False)
    test = Val_test(net, val_data)
    print ('start training')
    test.test_model()
