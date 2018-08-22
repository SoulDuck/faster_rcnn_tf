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

class RPN_eval(object):
    def __init__(self , model_path , feat_stride ,anchor_scales ):
        self.model_path = model_path
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.load_model()
    def load_model(self):
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(
            meta_graph_or_file=self.model_path + '.meta' ,)  # example model path ./models/fundus_300/5/model_1.ckpt
        self.saver.restore(self.sess, save_path=self.model_path)  # example model path ./models/fundus_300/5/model_1.ckpt
        self.top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
        self.cls_conv = tf.get_default_graph().get_tensor_by_name('cls/cls_output:0')
        self.bbox_conv = tf.get_default_graph().get_tensor_by_name('bbox/cls_output:0')

        self.rpn_labels_op = tf.get_default_graph().get_tensor_by_name('rpn_labels:0')
        self.rpn_bbox_targets_op = tf.get_default_graph().get_tensor_by_name('rpn_bbox_targets:0')
        self.rpn_bbox_inside_weights_op = tf.get_default_graph().get_tensor_by_name('rpn_bbox_inside_weights:0')
        self.rpn_bbox_outside_weights_op = tf.get_default_graph().get_tensor_by_name('rpn_bbox_outside_weights:0')

        self.indice_op = tf.get_default_graph().get_tensor_by_name('indice:0')
        self.bbox_targets_op =tf.get_default_graph().get_tensor_by_name('bbox_targets:0')
        self.bbox_inside_weights_op = tf.get_default_graph().get_tensor_by_name('bbox_inside_weights:0')
        self.bbox_outside_weights_op = tf.get_default_graph().get_tensor_by_name('bbox_outside_weights:0')

        self.x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
        self.im_dims = tf.get_default_graph().get_tensor_by_name('im_dims:0')
        self.gt_boxes = tf.get_default_graph().get_tensor_by_name('gt_boxes:0')
        self.phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        self.roi_blobs_op, self.roi_scores_op, self.roi_blobs_ori_op, self.roi_scores_ori_op, self.roi_softmax_op = \
            roi.roi_proposal(self.cls_conv, self.bbox_conv, self.im_dims, self.feat_stride, self.anchor_scales,
                             is_training=True)

    def load_simple_convnet(self , depth):
        for d in depth:
            self.w = tf.get_default_graph().get_tensor_by_name('conv_{}/W:0'.format(d))


    def eval(self , img ,src_gt_boxes ,src_im_dims ,_feat_stride , anchor_scales ):
        assert len(np.shape(img)) == 4 and np.shape(img)[0] == 1
        rpn_cls_score = np.zeros([1, int(math.ceil(h / float(_feat_stride))),
                                  int(math.ceil(w / float(_feat_stride))), 512])
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights = anchor_target(
            rpn_cls_score=rpn_cls_score, gt_boxes=src_gt_boxes, im_dims=src_im_dims, _feat_stride=_feat_stride,
            anchor_scales=anchor_scales)

        indices = np.where([np.reshape(rpn_labels, [-1]) > 0])[1]
        feed_dict = {self.x_: src_img, self.im_dims: src_im_dims, self.gt_boxes: src_gt_boxes, self.phase_train: True,
                     self.rpn_labels_op: rpn_labels,
                     self.bbox_targets_op: bbox_targets,
                     self.bbox_inside_weights_op: bbox_inside_weights,
                     self.bbox_outside_weights_op: bbox_outside_weights,
                     self.indice_op: indices}
        roi_blobs, roi_scores = self.sess.run([self.roi_blobs_op, self.roi_scores_op], feed_dict=feed_dict)
        # draw pos , neg rectangles and
        roi_scores = np.reshape(roi_scores , [-1,1])
        keep=non_maximum_supression(dets = np.hstack([roi_blobs , roi_scores]) , thresh=0.7)
        nms_roi_blobs = roi_blobs[keep]
        nms_roi_scores = roi_scores[keep]
        print 'NMS before {} ===> after {}'.format(len(roi_blobs) , len(nms_roi_blobs))
        draw_rectangles(src_img , nms_roi_blobs , nms_roi_scores ,src_gt_boxes , fname = '{}.png'.format('tmp'))

    def _prepare(self ,img ,src_gt_boxes ,src_im_dims ,_feat_stride , anchor_scales ):
        # save topconv caches
        assert len(np.shape(img)) == 4 and np.shape(img)[0] == 1
        rpn_cls_score = np.zeros([1, int(math.ceil(h / float(_feat_stride))),
                                  int(math.ceil(w / float(_feat_stride))), 512])
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights = anchor_target(
            rpn_cls_score=rpn_cls_score, gt_boxes=src_gt_boxes, im_dims=src_im_dims, _feat_stride=_feat_stride,
            anchor_scales=anchor_scales)

        indices = np.where([np.reshape(rpn_labels, [-1]) > 0])[1]
        feed_dict = {self.x_: src_img, self.im_dims: src_im_dims, self.gt_boxes: src_gt_boxes, self.phase_train: True,
                     self.rpn_labels_op: rpn_labels,
                     self.bbox_targets_op: bbox_targets,
                     self.bbox_inside_weights_op: bbox_inside_weights,
                     self.bbox_outside_weights_op: bbox_outside_weights,
                     self.indice_op: indices}
        return feed_dict

    def topconv_cache(self , img ,src_gt_boxes ,src_im_dims ,_feat_stride , anchor_scales , savepath):
        # save topconv caches
        feed_dict=self._prepare(img ,src_gt_boxes ,src_im_dims ,_feat_stride , anchor_scales)
        top_conv = self.sess.run([self.top_conv], feed_dict=feed_dict)
        top_conv = np.reshape(top_conv, list(np.shape(top_conv))[1:])
        np.save(file=savepath , arr = top_conv)

    def save_caches(self ,img_dir , gt_dir , anchor_scales  ,  savedir , img_ext ,gt_ext , func):
        # Get image paths
        img_paths = glob.glob(os.path.join(img_dir , '*.{}'.format(img_ext)))
        print img_paths
        for i,img_path in enumerate(img_paths) :
            progress(i , len(img_paths))
            name =  get_name(img_path)
            gt_path = os.path.join(gt_dir , name+'.{}'.format(gt_ext))
            # get Image , gt bboxes
            src_img = read_image(img_path) / 255.
            n,h,w,ch=np.shape(src_img)
            src_dims = [[h,w]]
            src_gtbboxes = read_gtbboxes(gt_path)
            savepath = os.path.join(savedir , name)
            func(src_img, src_gtbboxes, src_im_dims=src_dims, _feat_stride=self.feat_stride,
                               anchor_scales=anchor_scales, savepath=savepath)

    def rpn_cls_cache(self , img ,src_gt_boxes ,src_im_dims ,_feat_stride , anchor_scales , savepath):
        feed_dict = self._prepare(img, src_gt_boxes, src_im_dims, _feat_stride, anchor_scales)
        cls_conv = self.sess.run([self.cls_conv], feed_dict=feed_dict)
        cls_conv=np.reshape(cls_conv , np.shape(cls_conv)[1:])
        np.save(file=savepath , arr = cls_conv)

    def rpn_bbox_cache(self , img ,src_gt_boxes ,src_im_dims ,_feat_stride , anchor_scales , savepath):
        feed_dict = self._prepare(img, src_gt_boxes, src_im_dims, _feat_stride, anchor_scales)
        bbox_conv = self.sess.run([self.bbox_conv], feed_dict=feed_dict)
        bbox_conv=np.reshape(bbox_conv , np.shape(bbox_conv)[1:])
        np.save(file=savepath , arr = bbox_conv)



if __name__ == '__main__':
    # Get Image and Gt bbox
    img_path = './clutteredKIA/Images/frame0.jpg'
    gtbboxes_path = './clutteredKIA/Annotations/frame0.txt'
    src_img = read_image(img_path)/255.
    n,h,w,ch=np.shape(src_img)
    src_gtbboxes = read_gtbboxes(gtbboxes_path)
    # RPN CLASS
    rpn_eval=RPN_eval('rpn_models/model-24735' ,64 , anchor_scales = [24, 36, 50])
    # rpn cls
    #rpn_eval.rpn_cls_cache(src_img, src_gtbboxes, [[h, w]], 64, anchor_scales=[24, 36, 50], savepath='rpn_cls_caches')
    #rpn_eval.eval(src_img , src_gtbboxes, [[h,w]], 64 , anchor_scales = [24, 36, 50])
    #rpn_eval.topconv_cache(src_img, src_gtbboxes, [[h, w]], 64, anchor_scales=[24, 36, 50],
    #                        savepath='topconv_caches/frame0')

    img_dir = './clutteredKIA/Images'
    gt_dir = './clutteredKIA/Annotations'
    # Get TopConv
    rpn_eval.save_caches(img_dir, gt_dir, [24, 36, 50], savedir='topconv_caches', img_ext='jpg',
                                 gt_ext='txt' , func=rpn_eval.topconv_cache)
    # Get RPN cls conv
    rpn_eval.save_caches(img_dir, gt_dir, [24, 36, 50], savedir='rpn_cls_caches', img_ext='jpg',
                                 gt_ext='txt' , func=rpn_eval.rpn_cls_cache)
    # Get RPN bbox conv
    rpn_eval.save_caches(img_dir, gt_dir, [24, 36, 50], savedir='rpn_bbox_caches', img_ext='jpg',
                                     gt_ext='txt' , func=rpn_eval.rpn_bbox_cache)
