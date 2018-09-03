#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from utils import next_img_gtboxes , tracking_loss , non_maximum_supression , draw_fr_bboxes
from anchor_target_layer import anchor_target
from convnet import define_placeholder , simple_convnet , rpn_cls_layer , rpn_bbox_layer , sess_start , optimizer ,rpn_cls_loss , rpn_bbox_loss ,bbox_loss
from proposal_layer import inv_transform_layer , inv_transform_layer_fastrcnn
from proposal_target_layer import proposal_target_layer
from fast_rcnn import fast_rcnn , fast_rcnn_bbox_loss , fast_rcnn_cls_loss , nms_fast_rcnn_blobs , get_interest_target
import math
import roi
import sys , time
from mAP import poc_acc

rpn_labels_op = tf.placeholder(dtype =tf.int32 , shape=[1,1,None,None] , name = 'rpn_labels_op')
rpn_bbox_targets_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None] , name='rpn_bbox_targets_op')
rpn_bbox_inside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None] , name='rpn_bbox_inside_weights_op')
rpn_bbox_outside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None] , name='rpn_bbox_outside_weights_op')
indice_op = tf.placeholder(dtype =tf.int32 , shape=[None] , name='indice_op')
bbox_targets_op = tf.placeholder(dtype =tf.float32 , shape=[None,4] , name='bbox_targets_op')
bbox_inside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[None,4] , name='bbox_inside_weights_op')
bbox_outside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[None,4] , name='bbox_outside_weights_op')
x_ = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='x_')
im_dims = tf.placeholder(tf.int32, [None, 2], name='im_dims')
gt_boxes = tf.placeholder(tf.int32, [None, 5], name='gt_boxes')
phase_train = tf.placeholder(tf.bool, name='phase_train')
feat_stride_ = tf.placeholder(tf.int32, name='feat_stride')
anchor_scales_ = tf.placeholder(tf.int32 , [3], name='anchor_scales')

n_classes = 8+1
top_conv, _feat_stride = simple_convnet(x_)
# RPN CLS
rpn_cls = rpn_cls_layer(top_conv)
# RPN BBOX
rpn_bbox_pred = rpn_bbox_layer(top_conv)

rpn_cls_loss_op ,A_op ,B_op  = rpn_cls_loss(rpn_cls , rpn_labels_op) # rpn_labels_op 1 ,1 h ,w
rpn_cls_loss_op = tf.identity(rpn_cls_loss_op, name = 'rpn_cls_loss_op')

rpn_bbox_loss_op , diff_op , C_op , D_op ,E_op ,F_op= \
    bbox_loss(rpn_bbox_pred ,bbox_targets_op , bbox_inside_weights_op , bbox_outside_weights_op , rpn_labels_op)
rpn_bbox_loss_op = tf.identity(rpn_bbox_loss_op, name = 'rpn_bbox_loss_op')

anchor_scales = [24, 36, 50]

# BBOX OP
# INV inv_blobs_op OP = return to the original Coordinate
# INV target_inv_blobs_op OP = return to the original Coordinate (indices )
#inv_blobs_op  , target_inv_blobs_op = inv_transform_layer(rpn_bbox_pred ,  cfg_key = phase_train , \
#                                        _feat_stride = _feat_stride , anchor_scales =anchor_scales , indices = indice_op)
#inv_blobs_op=tf.identity(inv_blobs_op,name='inv_blobs_op')
# Region of Interested
roi_blobs_op, roi_scores_op , roi_blobs_ori_op ,roi_scores_ori_op  , roi_softmax_op = \
    roi.roi_proposal(rpn_cls , rpn_bbox_pred , im_dims , feat_stride_ , anchor_scales_ ,is_training=phase_train)
# NMS 후 >0.5 인것 fast rcnn 으로 넘기기
# Fast rcnn 을 학습시킬 roi을 추출합니다
ptl_rois_op, ptl_labels_op, ptl_bbox_targets_op, ptl_bbox_inside_weights_op, ptl_bbox_outside_weights_op = \
    proposal_target_layer(roi_blobs_op , gt_boxes , _num_classes= n_classes ) # ptl = Proposal Target Layer
# Fast RCNN LOGITS
# foreground 만 보내면 된다.
rois_op = tf.cond(phase_train , lambda : ptl_rois_op , lambda : roi_blobs_op)
fast_rcnn_cls_logits , fast_rcnn_bbox_logits = \
    fast_rcnn(top_conv , ptl_rois_op ,roi_blobs_op ,im_dims  , num_classes=n_classes , phase_train = phase_train)
fast_rcnn_cls_logits = tf.identity(fast_rcnn_cls_logits , 'fr_cls_logits_op')
fast_rcnn_bbox_logits = tf.identity(fast_rcnn_bbox_logits , 'fr_bboxes_logits_op')
# 만약 training 이면 ptl rois op 을 , eval 이면 roi_blobs_op 을 선택한다
# fast_rcnn_cls_logits 은 행이 4 * n_classes 만큼 나온다 . 그래서 cls 에 해당하는 좌표만 가져온다
# itr ==> interest
itr_fr_bbox_target_op = get_interest_target(tf.argmax(fast_rcnn_cls_logits , axis =1), fast_rcnn_bbox_logits , n_classes )
# fast rcnn 을 복원 시킨다
# fast rcnn blobs op shape : 1, N , 4 *n_classes
#
itr_fr_blobs_op = inv_transform_layer_fastrcnn( rois_op, itr_fr_bbox_target_op  )
# inv_ptl_rois_op = inv_transform_layer_fastrcnn(ptl_rois_op , ptl_bbox_targets_op)
# 임시로 되돌려 만들어 본다 , ptl 은 추천된 박스 영상과 동일해야 한다.
# FAST RCNN COST
fr_cls_loss_op = fast_rcnn_cls_loss(fast_rcnn_cls_logits , ptl_labels_op)
fr_bbox_loss_op = fast_rcnn_bbox_loss(fast_rcnn_bbox_logits ,ptl_bbox_targets_op , ptl_bbox_inside_weights_op , ptl_bbox_outside_weights_op )
# Train op and Loss op
rpn_cost_op = rpn_cls_loss_op + rpn_bbox_loss_op
fr_cost_op = fr_cls_loss_op + fr_bbox_loss_op
cost_op = rpn_cost_op  + fr_cost_op
train_op = optimizer(cost_op  , lr=0.0001)
# Start Session
sess=sess_start()
# Saver
saver = tf.train.Saver(max_to_keep=10)
# Write log
tb_writer = tf.summary.FileWriter('logs')
tb_writer.add_graph(tf.get_default_graph())
#
max_iter = 485 * 3000
max_acc = 0
# start Training
start_time = time.time()

for i in range(0, max_iter):

    src_img , src_gt_boxes = next_img_gtboxes(i)
    h,w,ch = np.shape(src_img)
    src_im_dims = [(h,w)]
    # ** GET RPN **
    rpn_cls_score = np.zeros([1, int(math.ceil(h / float(_feat_stride))),
                              int(math.ceil(w / float(_feat_stride))), 512])
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights = anchor_target(
        rpn_cls_score=rpn_cls_score, gt_boxes=src_gt_boxes, im_dims=src_im_dims, _feat_stride=_feat_stride,
        anchor_scales=anchor_scales)

    indices=np.where([np.reshape(rpn_labels,[-1])>0])[1]
    src_img=src_img.reshape([1]+list(np.shape(src_img)))
    if i % 16500 == 0 :
        feed_dict = {x_: src_img, im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: False,
                     rpn_labels_op: rpn_labels,
                     bbox_targets_op: bbox_targets,
                     bbox_inside_weights_op: bbox_inside_weights,
                     bbox_outside_weights_op: bbox_outside_weights,
                     indice_op: indices ,
                     feat_stride_ :  _feat_stride,
                     anchor_scales_ : anchor_scales }
        # Get Fast rcnn , RPN
        fr_cls , fr_bbox , roi_cls , roi_bbox ,itr_fr_blobs = sess.run(
            [fast_rcnn_cls_logits , fast_rcnn_bbox_logits , roi_scores_op ,roi_blobs_op ,itr_fr_blobs_op ] , feed_dict)
        #
        itr_fr_blobs = np.squeeze(itr_fr_blobs)
        fr_cls = np.argmax(fr_cls , axis =1 ).reshape([-1,1])
        fr_blobs_cls = np.hstack([itr_fr_blobs, fr_cls])
        nms_keep = non_maximum_supression(fr_blobs_cls ,0.5)
        print 'before nms {} ==> after nms {}'.format(len(fr_blobs_cls) , len(nms_keep ))
        nms_itr_fr_blobs=itr_fr_blobs[nms_keep]
        nms_fr_cls = fr_cls[nms_keep]
        acc = poc_acc(nms_itr_fr_blobs,nms_fr_cls , src_gt_boxes , 0.5)
        # model save
        saver.save(sess , save_path = 'models/{}'.format(i) , global_step= i)
        # save box
        # fast bbox 중에 foreground 만 보여준다
        draw_fr_bboxes(src_img , nms_fr_cls , nms_itr_fr_blobs, (255,0,0) , 3 ,savepath = 'result_fastrcnn_roi/{}.png'.format(i) )
        rpn_bbox , =sess.run([rpn_bbox_pred], feed_dict)
        #
# Training
    feed_dict = {x_: src_img, im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: True,
                 rpn_labels_op: rpn_labels,
                 bbox_targets_op: bbox_targets ,
                 bbox_inside_weights_op: bbox_inside_weights ,
                 bbox_outside_weights_op : bbox_outside_weights,
                 indice_op:indices,
                 feat_stride_: _feat_stride,
                 anchor_scales_: anchor_scales}

    #
    rois = sess.run(fetches=[rois_op], feed_dict=feed_dict)
    _ = sess.run(fetches=[train_op], feed_dict=feed_dict)
    fr_cls, fr_bbox = sess.run(
        [fast_rcnn_cls_logits, fast_rcnn_bbox_logits], feed_dict)
    sys.stdout.write('\r Progress {} / {}'.format(i,max_iter))
    sys.stdout.flush()
    # Get Loss value
    # Write log
    rpn_cls_loss, rpn_bbox_loss, fr_cls_loss, fr_bbox_loss = sess.run(
        [rpn_cls_loss_op, rpn_bbox_loss_op, fr_cls_loss_op, fr_bbox_loss_op], feed_dict)
    tracking_loss(tb_writer, rpn_cls_loss, i, 'rpn cls loss')
    tracking_loss(tb_writer, rpn_bbox_loss, i, 'rpn bbox loss')
    tracking_loss(tb_writer, fr_cls_loss, i, 'fast rcnn cls loss')
    tracking_loss(tb_writer, fr_bbox_loss, i, 'fast rcnn bbox loss')

