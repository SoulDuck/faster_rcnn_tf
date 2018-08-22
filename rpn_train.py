# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from utils import next_img_gtboxes, draw_rectangles, non_maximum_supression, draw_rectangles_fastrcnn
from anchor_target_layer import anchor_target
from convnet import define_placeholder, simple_convnet, rpn_cls_layer, rpn_bbox_layer, sess_start, optimizer, \
    rpn_cls_loss, rpn_bbox_loss, bbox_loss
from proposal_layer import inv_transform_layer, inv_transform_layer_fastrcnn
from proposal_target_layer import proposal_target_layer
from fast_rcnn import fast_rcnn, fast_rcnn_bbox_loss, fast_rcnn_cls_loss
import math
import roi
import sys

rpn_labels_op = tf.placeholder(dtype=tf.int32, shape=[1, 1, None, None])
rpn_bbox_targets_op = tf.placeholder(dtype=tf.float32, shape=[1, 36, None, None])
rpn_bbox_inside_weights_op = tf.placeholder(dtype=tf.float32, shape=[1, 36, None, None])
rpn_bbox_outside_weights_op = tf.placeholder(dtype=tf.float32, shape=[1, 36, None, None])
indice_op = tf.placeholder(dtype=tf.int32, shape=[None])
bbox_targets_op = tf.placeholder(dtype=tf.float32, shape=[None, 4])
bbox_inside_weights_op = tf.placeholder(dtype=tf.float32, shape=[None, 4])
bbox_outside_weights_op = tf.placeholder(dtype=tf.float32, shape=[None, 4])
x_, im_dims, gt_boxes, phase_train = define_placeholder()
n_classes = 10 + 1
top_conv, _feat_stride = simple_convnet(x_)
# RPN CLS
rpn_cls = rpn_cls_layer(top_conv)
# RPN BBOX
rpn_bbox_pred = rpn_bbox_layer(top_conv)
# Fast_rcnn(top_conv ,rois , im_dims , eval_mode=False , num_classes=10 , phase_train = phase_train)

# CLS LOSS
# A_op : rpn cls pred
# B_op : binary_indices
# B_1_op : indices
rpn_cls_loss_op, A_op, B_op = rpn_cls_loss(rpn_cls, rpn_labels_op)  # rpn_labels_op 1 ,1 h ,w

# BBOX LOSS
# C_op : indiced rpn bbox  pred op
# D_op : indiced rpn target  op
# E_op : rpn_inside_weights
# F_op : rpn_outside_weights
rpn_bbox_loss_op, diff_op, C_op, D_op, E_op, F_op = \
    bbox_loss(rpn_bbox_pred, bbox_targets_op, bbox_inside_weights_op, bbox_outside_weights_op, rpn_labels_op)
anchor_scales = [3, 4, 5]
# BBOX OP
# INV inv_blobs_op OP = return to the original Coordinate
# INV target_inv_blobs_op OP = return to the original Coordinate (indices )
inv_blobs_op, target_inv_blobs_op = inv_transform_layer(rpn_bbox_pred, cfg_key=phase_train, \
                                                        _feat_stride=_feat_stride, anchor_scales=anchor_scales,
                                                        indices=indice_op)

# Region of Interested
roi_blobs_op, roi_scores_op, roi_blobs_ori_op, roi_scores_ori_op, roi_softmax_op = \
    roi.roi_proposal(rpn_cls, rpn_bbox_pred, im_dims, _feat_stride, anchor_scales, is_training=True)
# Fast rcnn 을 학습시킬 roi을 추출합니다
ptl_rois_op, ptl_labels_op, ptl_bbox_targets_op, ptl_bbox_inside_weights_op, ptl_bbox_outside_weights_op = \
    proposal_target_layer(roi_blobs_op, gt_boxes, _num_classes=n_classes)  # ptl = Proposal Target Layer


# Inverse target fast rcnn bbox
# RPN COST
rpn_cost_op = rpn_cls_loss_op + rpn_bbox_loss_op
cost_op = rpn_cost_op
train_op = optimizer(cost_op, lr=0.0001)
sess = sess_start()
max_iter = 55000 * 25

for i in range(2, max_iter):
    # Input Image
    src_img, src_gt_boxes = next_img_gtboxes(i)
    h, w = np.shape(src_img)[:2]
    src_im_dims = [(h, w)]

    # rpn labels RPN Network 을 학습시킬 Label 이 있다 foreground 이면 1 , backgound = 0 , 학습 시키지 않을 box에 대해서는 -1 을 부여한다
    # rpn_bbox_targets : anchor 들을 target 으로 변환했다
    # src im dims : source image 의 원래 크기
    # _feat stride : 얼마만큼 줄였는지
    # _anchor scale : anchor scale
    rpn_cls_score = np.zeros([1, int(math.ceil(h / float(_feat_stride))), int(math.ceil(w / float(_feat_stride))), 512])
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
        anchor_target(rpn_cls_score, src_gt_boxes, src_im_dims, _feat_stride, anchor_scales)


    indices = np.where([np.reshape(rpn_labels, [-1]) > 0])[1]

    src_img = src_img.reshape([1] + list(np.shape(src_img)) + [1])
    feed_dict = {x_: src_img, im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: True,
                 rpn_labels_op: rpn_labels,
                 bbox_targets_op: bbox_targets,
                 bbox_inside_weights_op: bbox_inside_weights,
                 bbox_outside_weights_op: bbox_outside_weights,
                 indice_op: indices}

    rpn_labels, cls_cost, bbox_cost, rpn_cls_value, A, B, diff, C, D, E, F, roi_blobs, roi_scores, target_inv_blobs = sess.run(
        fetches=[rpn_labels_op, rpn_cls_loss_op, rpn_bbox_loss_op, rpn_cls, A_op, B_op, diff_op, C_op, D_op, E_op, F_op,
                 roi_blobs_op, roi_scores_op, target_inv_blobs_op], feed_dict=feed_dict)
    roi_blobs_ori, roi_scores_ori, roi_softmax = sess.run(fetches=[roi_blobs_ori_op, roi_scores_ori_op, roi_softmax_op],
                                                          feed_dict=feed_dict)
    ptl_rois, ptl_labels, plt_bbox_targets, ptl_bbox_inside_weights, ptl_bbox_outside_weights = sess.run(
        fetches=[ptl_rois_op, ptl_labels_op, ptl_bbox_targets_op, ptl_bbox_inside_weights_op,
                 ptl_bbox_outside_weights_op], feed_dict=feed_dict)
    _ = sess.run(fetches=[train_op], feed_dict=feed_dict)
    pos_blobs = roi_blobs[np.where([roi_scores > 0.5])[1]]

    if i % 1000 == 0:
        print 'POS BBOX \t', pos_blobs
        print 'ROI SCORE \t', np.shape(roi_scores)
        print 'ROI BLOBS \t', np.shape(roi_blobs)
        print 'ROI POS BLOBS \t', np.shape(pos_blobs)
        print 'ANCHOR POS INDICE \t', indices

        print 'RPN CLS prediction ', A
        print 'indiced Pred bbox', C
        print 'indiced target bbox', D
        print 'inside Weight', E
        print 'outside Weight', F
        print 'indices binary ', B
        print 'rpn_labels_op', rpn_labels_op
        print 'target_inv_bbox ', target_inv_blobs
        print 'rpn_cls_value \t', np.shape(rpn_cls_value)

        """ RPN CLS 변환 """
        n, h, w, ch = rpn_cls_value.shape
        rpn_cls_value = rpn_cls_value.transpose([0, 3, 1, 2])
        rpn_cls_value = rpn_cls_value.reshape([1, 2, ch // 2 * h, w])
        rpn_cls_value = rpn_cls_value.transpose([0, 2, 3, 1])
        rpn_cls_value = rpn_cls_value.reshape([-1, 2])

        roi_softmax = roi_softmax.transpose([0, 3, 1, 2])
        roi_softmax = roi_softmax.reshape([1, 2, ch // 2 * h, w])
        roi_softmax = roi_softmax.transpose([0, 2, 3, 1])
        roi_softmax = roi_softmax.reshape([-1, 2])

        print 'POS rpn_cls_value PROB ', rpn_cls_value[indices]
        print 'POS SOFTMAX PROB , {}'.format(roi_softmax[indices])
        print 'ROI BBOX 에서 ANCHOR같은 indices 을 뽑은것 ', roi_blobs_ori[indices]
        print 'ROI CLS 에서 ANCHOR같은 indices 을 뽑은것', roi_scores_ori[indices]

        print 'RPN CLS LOSS : \t', cls_cost
        print 'RPN BBOX LOSS \t', bbox_cost

        savepath_anchor = './result_anchor/{}.png'.format(i)
        savepath_roi = './result_roi/{}.png'.format(i)
        src_img = np.squeeze(src_img)
        target_inv_blobs = target_inv_blobs.astype(np.int)
        # draw_rectangles(src_img, roi_blobs[:,1:], roi_scores, target_inv_blobs , savepath_roi ,color='r')
        """
        # 겹치는 영역하고 , NMS 을 적용하지 않은 ROI 
        pos_indices=np.where([roi_scores_ori>0.5])[1]
        # NMS
        roi_blobs 에 NMS 가 적용 되어 있다
        roi_Scores 는 NMS가 적용된 좌표에 대응하는 확률값이다
        target_inv_blobs 는 blobs을 
        """
        pos_indices = np.where([roi_scores_ori > 0.5])[1]
        print 'Positive : {}'.format(len(pos_indices))
        print np.shape(roi_blobs_ori)
        draw_rectangles(src_img, roi_blobs_ori[:, :], roi_scores_ori, target_inv_blobs, None, savepath_roi, color='r')
    sys.stdout.write('\r Progress {} {}'.format(i, max_iter))
    sys.stdout.flush()
