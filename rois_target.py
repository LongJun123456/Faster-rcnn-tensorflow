# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:20:04 2018

@author: LongJun
"""
import config as cfg
from anchor_label import calculate_IOU
import numpy as np
import numpy.random as npr
def proposal_target(rpn_rois, rpn_scores, gt_boxes, _num_classes, gt_cls):
    """ Fast/Faster rcnn proposal_target layer, used for computing the target of convergence"""
    all_rois = rpn_rois
    all_scores = rpn_scores
    num_images = 1
    rois_per_image = cfg.dect_train_batch/ num_images
    fg_rois_per_image = np.round(cfg.dect_fg_rate* rois_per_image)
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(\
            all_rois, all_scores, gt_boxes, fg_rois_per_image,\
            rois_per_image, _num_classes, gt_cls)
    rois = rois.reshape(-1, 5) 
    roi_scores = roi_scores.reshape(-1) 
    labels = labels.reshape(-1, 1) 
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4) 
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
 
    
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """ compute the bbox_targets and bbox_inside_weights
        ie, tx*,ty*,tw*,th* and which bbox_target to be used in loss compute"""
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32) 
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0] 
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.roi_input_inside_weight
    return bbox_targets,bbox_inside_weights
     
def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, gt_cls): 
    """ rois sample process:  clip to the image boundary, nms, bg_fg sample"""
    overlaps = calculate_IOU(all_rois[:, 1:5], gt_boxes)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_cls[gt_assignment]
    fg_inds = np.where(max_overlaps >= cfg.fg_thresh)[0]
    bg_inds = np.where((max_overlaps < cfg.bg_thresh_hi) &(max_overlaps >= cfg.bg_thresh_lo))[0]
    #print(np.sum(fg_inds), np.sum(bg_inds))
    if fg_inds.size > 0 and bg_inds.size > 0: 
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size) 
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False) 
        bg_rois_per_image = rois_per_image - fg_rois_per_image 
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace) 
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    else:
        import pdb
        pdb.set_trace()
    keep_inds = np.append(fg_inds, bg_inds)
    labels = labels[keep_inds] 
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]
    
     
    bbox_target_data = _compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :], labels)
    
   
    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)
    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois) 
  if cfg.bbox_nor_target_pre:
   
    targets = ((targets - np.array(cfg.bbox_nor_mean))/np.array(cfg.bbox_nor_stdv)) 
  return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)
  
def bbox_transform(ex_rois, gt_rois):
  """ convert the coordinate of gt_rois into targets form using ex_rois """
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
  gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
  gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths 
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights 
  targets_dw = np.log(gt_widths / ex_widths)
  targets_dh = np.log(gt_heights / ex_heights)

  targets = np.vstack(
    (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
  return targets
    
