#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from utils import next_img_gtboxes , draw_rectangles , non_maximum_supression , draw_rectangles_fastrcnn , draw_rectangles_ptl , progress , tracking_rpn_loss
from utils import read_image , read_gtbboxes
from anchor_target_layer import anchor_target
from convnet import define_placeholder , simple_convnet , rpn_cls_layer , rpn_bbox_layer , sess_start , optimizer ,rpn_cls_loss , rpn_bbox_loss ,bbox_loss
from proposal_layer import inv_transform_layer , inv_transform_layer_fastrcnn
from proposal_target_layer import proposal_target_layer
from fast_rcnn import fast_rcnn , fast_rcnn_bbox_loss , fast_rcnn_cls_loss , nms_fast_rcnn_blobs
import math
import roi
import sys

rpn_labels_op = tf.placeholder(dtype =tf.int32 , shape=[1,1,None,None] , name =  'rpn_labels')
rpn_bbox_targets_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None] , name = 'rpn_bbox_targets' )
rpn_bbox_inside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None] , name='rpn_bbox_inside_weights')
rpn_bbox_outside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None] , name='rpn_bbox_outside_weights')

indice_op = tf.placeholder(dtype =tf.int32 , shape=[None] , name = 'indice')
bbox_targets_op = tf.placeholder(dtype =tf.float32 , shape=[None,4] , name='bbox_targets')
bbox_inside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[None,4] ,name='bbox_inside_weights')
bbox_outside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[None,4] , name ='bbox_outside_weights')
#x_, im_dims, gt_boxes, phase_train = define_placeholder()
x_ = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='x_')
im_dims = tf.placeholder(tf.int32, [None, 2], name='im_dims')
gt_boxes = tf.placeholder(tf.int32, [None, 5], name='gt_boxes')
phase_train = tf.placeholder(tf.bool, name='phase_train')


n_classes = 19+1
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
rpn_cls_loss_op ,A_op ,B_op  = rpn_cls_loss(rpn_cls , rpn_labels_op) # rpn_labels_op 1 ,1 h ,w
# BBOX LOSS
# C_op : indiced rpn bbox  pred op
# D_op : indiced rpn target  op
# E_op : rpn_inside_weights
# F_op : rpn_outside_weights
rpn_bbox_loss_op , diff_op , C_op , D_op ,E_op ,F_op= \
    bbox_loss(rpn_bbox_pred ,bbox_targets_op , bbox_inside_weights_op , bbox_outside_weights_op , rpn_labels_op)
anchor_scales = [ 24, 36, 50 ]
# BBOX OP
# INV inv_blobs_op OP = return to the original Coordinate
# INV target_inv_blobs_op OP = return to the original Coordinate (indices )
inv_blobs_op  , target_inv_blobs_op = inv_transform_layer(rpn_bbox_pred ,  cfg_key = phase_train , \
                                        _feat_stride = _feat_stride , anchor_scales =anchor_scales , indices = indice_op)
# Region of Interested
#2018 8.20 ==> nms 을 제거
roi_blobs_op, roi_scores_op , roi_blobs_ori_op ,roi_scores_ori_op  , roi_softmax_op = \
    roi.roi_proposal(rpn_cls , rpn_bbox_pred , im_dims , _feat_stride , anchor_scales ,is_training=True)
# Fast rcnn 을 학습시킬 roi을 추출합니다
ptl_rois_op, ptl_labels_op, ptl_bbox_targets_op, ptl_bbox_inside_weights_op, ptl_bbox_outside_weights_op = \
    proposal_target_layer(roi_blobs_op , gt_boxes , _num_classes= n_classes ) # ptl = Proposal Target Layer
inv_ptl_rois_op = inv_transform_layer_fastrcnn(ptl_rois_op , ptl_bbox_targets_op)

rpn_cost_op = rpn_cls_loss_op + rpn_bbox_loss_op
train_cls_op = optimizer(rpn_cls_loss_op , lr=0.01)
train_bbox_op = optimizer(rpn_bbox_loss_op , lr = 0.001)
cost_op = rpn_cost_op
# only rpn 학습

train_op = optimizer( cost_op , lr = 0.001 )
sess=sess_start()
max_iter = 485 * 100
saver = tf.train.Saver(max_to_keep=10)
tb_writer = tf.summary.FileWriter('rpn_logs')
tb_writer.add_graph(tf.get_default_graph())

f=open('tmp_log.txt','w')


for i in range(max_iter):
    progress(i,max_iter)
    src_img , src_gt_boxes = next_img_gtboxes(i)
    h,w,ch = np.shape(src_img)
    src_im_dims = [(h,w)]
    # rpn 구하는 코드
    rpn_cls_score = np.zeros([1, int(math.ceil(h / float(_feat_stride))),
                              int(math.ceil(w / float(_feat_stride))), 512])
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights = anchor_target(
        rpn_cls_score=rpn_cls_score, gt_boxes=src_gt_boxes, im_dims=src_im_dims, _feat_stride=_feat_stride,
        anchor_scales=anchor_scales)

    indices=np.where([np.reshape(rpn_labels,[-1])>0])[1]
    src_img=src_img.reshape([1]+list(np.shape(src_img)))
    feed_dict = {x_: src_img, im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: True,
                 rpn_labels_op: rpn_labels,
                 bbox_targets_op: bbox_targets ,
                 bbox_inside_weights_op: bbox_inside_weights ,
                 bbox_outside_weights_op : bbox_outside_weights,
                 indice_op:indices}

    if i % 485 ==0:
        """
        img_path = './clutteredKIA/Images/frame0.jpg'
        gtbboxes_path = './clutteredKIA/Annotations/frame0.txt'
        src_img = read_image(img_path)
        n, h, w, ch = np.shape(src_img)
        src_gt_boxes = read_gtbboxes(gtbboxes_path)
        indices = np.where([np.reshape(rpn_labels, [-1]) > 0])[1]
        feed_dict = {x_: src_img, im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: True,
                     rpn_labels_op: rpn_labels,
                     bbox_targets_op: bbox_targets,
                     bbox_inside_weights_op: bbox_inside_weights,
                     bbox_outside_weights_op: bbox_outside_weights,
                     indice_op: indices}
        """

        # roi blobs and scores
        roi_blobs, roi_scores, rpn_loss = sess.run([roi_blobs_op, roi_scores_op, cost_op], feed_dict=feed_dict)
        cls_values , bbox_values = sess.run([rpn_cls, rpn_bbox_pred], feed_dict=feed_dict)
        # draw pos , neg rectangles and
        draw_rectangles(src_img , roi_blobs , roi_scores ,src_gt_boxes , fname = '{}.png'.format(i) )

        # save model
        saver.save(sess, save_path='rpn_models/model', global_step=i)
        # write rpn train loss
        print '\t',rpn_loss
        # write train RPN loss
        tracking_rpn_loss(tb_writer, rpn_loss, i)
        # write tmp log
        f.write('n roi_blobs : {}  , roi_blobs mean {} , roi_scores mean {}'.format(len(roi_blobs), np.mean(roi_blobs),
                np.mean(roi_scores)))
    # Training
    _ = sess.run([train_op], feed_dict=feed_dict)
    f.flush()
f.close()


    

