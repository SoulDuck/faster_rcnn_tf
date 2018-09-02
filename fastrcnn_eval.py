#-*- coding:utf-8 -*-
import tensorflow as tf
#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
from anchor_target_layer import anchor_target
import math
from utils import read_gtbboxes , read_image , draw_rectangles , get_name
import roi
from utils import next_img_gtboxes , non_maximum_supression , progress
from proposal_layer import inv_transform_layer , inv_transform_layer_fastrcnn
import glob , os
from proposal_target_layer import proposal_target_layer
from fast_rcnn import fast_rcnn , get_interest_target
from utils import draw_fr_bboxes ,next_img_gtboxes_with_path

model_path = 'models/759000-759000'
sess = tf.Session()
saver = tf.train.import_meta_graph(
    meta_graph_or_file=model_path + '.meta', )  # example model path ./models/fundus_300/5/model_1.ckpt
saver.restore(sess, save_path=model_path)  # example model path ./models/fundus_300/5/model_1.ckpt
rpn_label_op=tf.get_default_graph().get_tensor_by_name('rpn_labels_op:0')
rpn_bbox_targets_op=tf.get_default_graph().get_tensor_by_name('rpn_bbox_targets_op:0')
rpn_bbox_inside_weights_op=tf.get_default_graph().get_tensor_by_name('rpn_bbox_inside_weights_op:0')
rpn_bbox_outside_weights_op=tf.get_default_graph().get_tensor_by_name('rpn_bbox_outside_weights_op:0')

indice_op=tf.get_default_graph().get_tensor_by_name('indice_op:0')
bbox_targets_op=tf.get_default_graph().get_tensor_by_name('bbox_targets_op:0')
bbox_inside_weights_op=tf.get_default_graph().get_tensor_by_name('bbox_inside_weights_op:0')
bbox_outside_weights_op=tf.get_default_graph().get_tensor_by_name('bbox_outside_weights_op:0')

x_=tf.get_default_graph().get_tensor_by_name('x_:0')
im_dims=tf.get_default_graph().get_tensor_by_name('im_dims:0')
gt_boxes=tf.get_default_graph().get_tensor_by_name('gt_boxes:0')
phase_train=tf.get_default_graph().get_tensor_by_name('phase_train:0')
feat_stride_ = tf.get_default_graph().get_tensor_by_name('feat_stride:0')
anchor_scales_ = tf.get_default_graph().get_tensor_by_name('anchor_scales:0')


roi_blobs_op = tf.get_default_graph().get_tensor_by_name('roi_blobs_op:0')
#inv_blobs_op = tf.get_default_graph().get_tensor_by_name('inv_blobs_op:0')

fr_cls_op = tf.get_default_graph().get_tensor_by_name('fr_cls_logits_op:0')
fr_bboxes_op = tf.get_default_graph().get_tensor_by_name('fr_bboxes_logits_op:0')
cls_output = tf.get_default_graph().get_tensor_by_name('cls/cls_output:0')



rpn_bbox_pred = tf.get_default_graph().get_tensor_by_name('bbox/cls_output:0')
rpn_cls = tf.get_default_graph().get_tensor_by_name('cls/cls_output:0')
rpn_cls_loss_op = tf.get_default_graph().get_tensor_by_name('rpn_cls_loss_op:0')
rpn_bbox_loss_op = tf.get_default_graph().get_tensor_by_name('rpn_bbox_loss_op:0')

n_classes = 8+1
roi_blobs_op, roi_scores_op , roi_blobs_ori_op ,roi_scores_ori_op  , roi_softmax_op = \
    roi.roi_proposal(rpn_cls , rpn_bbox_pred , im_dims , feat_stride_ , anchor_scales_ ,is_training=phase_train)
ptl_rois_op, ptl_labels_op, ptl_bbox_targets_op, ptl_bbox_inside_weights_op, ptl_bbox_outside_weights_op = \
    proposal_target_layer(roi_blobs_op , gt_boxes , _num_classes= n_classes ) # ptl = Proposal Target Layer

# reference
# https://stackoverflow.com/questions/43644506/setting-up-py-func-op-after-importing-graphdef-in-tensorflow
itr_fr_bbox_target_op = get_interest_target(tf.argmax(fr_cls_op , axis =1), fr_bboxes_op , n_classes )
#
itr_fr_blobs_op = inv_transform_layer_fastrcnn( roi_blobs_op, itr_fr_bbox_target_op  )
#



if __name__ == '__main__':
    for i in range(4058):
        src_img , src_gt_boxes , path  = next_img_gtboxes_with_path(i)
        name=os.path.split(path)[-1]

        h, w, ch = np.shape(src_img)
        src_im_dims = [(h, w)]
        src_img = src_img.reshape([1] + list(np.shape(src_img)))
        src_img = src_img
        #
        anchor_scales = [24, 36, 50]
        strides = [2, 2, 2, 2, 2, 2]
        _feat_stride = np.prod(strides)
        rpn_cls_score = np.zeros([1, int(math.ceil(h / float(_feat_stride))),
                                  int(math.ceil(w / float(_feat_stride))), 512])
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights = anchor_target(
            rpn_cls_score=rpn_cls_score, gt_boxes=src_gt_boxes, im_dims=src_im_dims, _feat_stride=_feat_stride,
            anchor_scales=anchor_scales)
        indices=np.where([np.reshape(rpn_labels,[-1])>0])[1]
        feed_dict = {x_: src_img, im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: False , feat_stride_ : _feat_stride , anchor_scales_ : anchor_scales ,
                     rpn_label_op: rpn_labels ,  rpn_bbox_targets_op : rpn_bbox_targets , rpn_bbox_inside_weights_op : rpn_bbox_inside_weights ,
                     rpn_bbox_outside_weights_op : rpn_bbox_outside_weights ,indice_op: indices  ,
                     bbox_targets_op : bbox_targets , bbox_inside_weights_op : bbox_inside_weights , bbox_outside_weights_op : bbox_outside_weights}
        fetches = [ fr_bboxes_op , fr_cls_op  ,itr_fr_blobs_op ]#inv_blobs_op
        fr_bboxes , fr_cls , itr_fr_blobs= sess.run(fetches , feed_dict)
        itr_fr_blobs=np.squeeze(itr_fr_blobs)
        print itr_fr_blobs[:10]
        fr_cls = np.argmax(fr_cls, axis=1).reshape([-1, 1])
        fr_blobs_cls = np.hstack([itr_fr_blobs, fr_cls])
        nms_keep = non_maximum_supression(fr_blobs_cls, 0.5)
        print 'before nms {} ==> after nms {}'.format(len(fr_blobs_cls), len(nms_keep))
        nms_itr_fr_blobs = itr_fr_blobs[nms_keep]
        nms_fr_cls = fr_cls[nms_keep]
        draw_fr_bboxes(src_img, nms_fr_cls, nms_itr_fr_blobs, (255, 0, 0), 3,
                       savepath='result_test_images/{}'.format(name))
"""

[[ 1211.35754395   209.34394836  1279.62561035   291.44747925]
 [ 1211.91479492   271.81689453  1280.19873047   355.66577148]
 [  331.9508667    335.83569336   387.56140137   417.71887207]
 [ 1212.45776367   336.54882812  1280.85229492   418.35437012]
 [ 1209.86730957    18.80626678  1279.69226074   101.37349701]
 [ 1212.03894043   593.04943848  1280.52062988   674.54626465]
 [ 1212.05285645   528.96557617  1280.71594238   610.48461914]
 [ 1212.5267334    401.09655762  1280.98937988   482.75634766]
 [ 1211.48852539   145.05401611  1279.69262695   226.50927734]
 [ 1163.296875     208.92800903  1219.38916016   291.55599976]]
 
"""
