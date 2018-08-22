from rpn_eval import RPN_eval
from fast_rcnn import fast_rcnn , fast_rcnn_bbox_loss , fast_rcnn_cls_loss , bbox_transform_inv ,get_interest_target
from proposal_target_layer import proposal_target_layer
from utils import read_image , read_gtbboxes ,get_name , draw_rectangles_fastrcnn
from anchor_target_layer import anchor_target
from proposal_layer import inv_transform_layer_fastrcnn
import math
import numpy as np
import tensorflow as tf
import roi , glob , os
# Settings placeholder and params
rpn_eval = RPN_eval('rpn_models/model-24735', 64, anchor_scales=[24, 36, 50])
n_classes = 19 + 1
_feat_stride = rpn_eval.feat_stride
anchor_scales=rpn_eval.anchor_scales
# From RPN

conv_h , conv_w  , conv_out = 12 ,20 , 128
n_anchor = 9
top_conv_op = tf.placeholder(dtype = tf.float32 ,shape = [1,conv_h,conv_w,conv_out] ,name= 'top_conv_op')
roi_cls_op = tf.placeholder(dtype = tf.float32  , shape = [1,conv_h,conv_w,n_anchor * 2] , name = 'rpn_cls_op')
roi_bbox_op = tf.placeholder(dtype = tf.float32 , shape = [1,conv_h,conv_w,n_anchor * 4] , name = 'rpn_bbox_op')
#
im_dims = tf.placeholder(tf.int32, [None, 2], name='im_dims')
gt_boxes = tf.placeholder(tf.int32, [None, 5], name='gt_boxes')
phase_train = tf.placeholder(tf.bool, name='phase_train')

# Region of Interested
roi_blobs_op, roi_scores_op , roi_blobs_ori_op ,roi_scores_ori_op  , roi_softmax_op = \
    roi.roi_proposal(roi_cls_op , roi_bbox_op  , im_dims , _feat_stride , anchor_scales ,is_training=True)
# Proposoal Target Layer
ptl_rois_op, ptl_labels_op, ptl_bbox_targets_op, ptl_bbox_inside_weights_op, ptl_bbox_outside_weights_op = \
    proposal_target_layer(roi_blobs_op , gt_boxes , _num_classes= n_classes ) # ptl = Proposal Target Layer
# Fast RCNN Model
fast_rcnn_cls_logits , fast_rcnn_bbox_logits = \
    fast_rcnn(top_conv_op , ptl_rois_op , im_dims , eval_mode=False , num_classes=n_classes , phase_train = phase_train)

fr_cls_loss_op = fast_rcnn_cls_loss(fast_rcnn_cls_logits , ptl_labels_op)
fr_bbox_loss_op = fast_rcnn_bbox_loss(fast_rcnn_bbox_logits ,ptl_bbox_targets_op , ptl_bbox_inside_weights_op , ptl_bbox_outside_weights_op )
fr_cost_op = fr_cls_loss_op  + fr_bbox_loss_op

lr=0.0001
train_op= tf.train.AdamOptimizer(learning_rate=lr).minimize(fr_cost_op )
sess=tf.Session()
init=tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)

#feed dict
if __name__ == '__main__':
    paths=glob.glob('rpn_bbox_caches/*.npy')
    names=map(lambda path : get_name(path) , paths)
    # read Image
    img_path = './clutteredKIA/Images/frame0.jpg'
    src_img = read_image(img_path) / 255.
    n, h, w, ch = np.shape(src_img)
    src_im_dims = [[h, w]]
    # read gt bboxes
    gtbboxes_path = './clutteredKIA/Annotations/frame0.txt'
    src_gt_boxes = read_gtbboxes(gtbboxes_path)
    # Get RPN cls , bbox
    assert len(np.shape(src_img)) == 4 and np.shape(src_img)[0] == 1
    for i,name in enumerate(names*100):
        roi_bbox =np.load('rpn_bbox_caches/{}.npy'.format(name))
        roi_cls = np.load('rpn_cls_caches/{}.npy'.format(name))
        top_conv = np.load('topconv_caches/{}.npy'.format(name))


        rpn_cls_score = np.zeros([1, int(math.ceil(h / float(_feat_stride))),
                                  int(math.ceil(w / float(_feat_stride))), 512])
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights = anchor_target(
            rpn_cls_score=rpn_cls_score, gt_boxes=src_gt_boxes, im_dims=src_im_dims, _feat_stride=_feat_stride,
            anchor_scales=anchor_scales)

        indices = np.where([np.reshape(rpn_labels, [-1]) > 0])[1]
        feed_dict = {top_conv_op: top_conv, roi_cls_op  : roi_cls ,roi_bbox_op :roi_bbox ,
                     im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: True}
        _ , fr_cost = sess.run([train_op , fr_cost_op],feed_dict)
        fast_rcnn_cls, ptl_labels, ptl_rois, ptl_targets = sess.run(
            [fast_rcnn_cls_logits, ptl_labels_op, ptl_rois_op, ptl_bbox_targets_op], feed_dict)
        print np.argmax(fast_rcnn_cls , axis=1)
        print ptl_labels
        """
        #  Get Proposal Target Layer Values
        ptl_labels, ptl_rois, ptl_targets = sess.run([fast_rcnn_cls_logits,ptl_labels_op, ptl_rois_op, ptl_bbox_targets_op], feed_dict)
        fr_cls_logits, fr_bbox_logits ,fr_cls_loss, fr_bbox_loss = sess.run(
            [fast_rcnn_cls_logits, fast_rcnn_bbox_logits, fr_cls_loss_op, fr_bbox_loss_op], feed_dict)
        
        print ptl_labels
        print np.argmax(fr_cls_logits , axis =0)
        

        """
        """
        #] Get Proposal Target Layer Values
        ptl_labels , ptl_rois , ptl_targets = sess.run([ptl_labels_op  ,ptl_rois_op ,ptl_bbox_targets_op], feed_dict)
        # Get FastRCNN Values
        fr_cls_logits, fr_bbox_logits ,fr_cls_loss, fr_bbox_loss = sess.run(
            [fast_rcnn_cls_logits, fast_rcnn_bbox_logits, fr_cls_loss_op, fr_bbox_loss_op], feed_dict)
        # fast rcnn
        fr_cls_logits = np.argmax( fr_cls_logits , axis=1)
        target_fr_bboxes = get_interest_target(fr_cls_logits , fr_bbox_logits, n_classes)
        fr_blobs = bbox_transform_inv(ptl_rois[:,1:] , target_fr_bboxes)
        # ptl bbox
        target_ptl_bboxes = get_interest_target(ptl_labels, ptl_targets, n_classes)
        ptl_blobs = bbox_transform_inv(ptl_rois[:, 1:], target_ptl_bboxes)
        #
        savepath = os.path.join('result_fr_bbox', name + '.jpg')
        src_img = np.squeeze(src_img)
        draw_rectangles_fastrcnn(src_img ,fr_blobs , true_classes=fr_cls_logits , savepath = savepath)
        acc = np.sum(np.equal(ptl_labels.astype(np.int32),fr_cls_logits)) / float(
            len(fr_cls_logits))
        if i % 100 ==0:
            print ptl_labels
            print fr_cls_logits
            print '{} = cls : {} + bbox :{} '.format(fr_cls_loss + fr_bbox_loss , fr_cls_loss ,fr_bbox_loss)

        """