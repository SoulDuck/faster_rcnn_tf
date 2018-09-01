import configure as cfg
import tensorflow as tf
from utils import non_maximum_supression
from cnn import affine ,dropout
import tensorflow as tf
import numpy as np

def smoothL1(x, sigma):
    '''
    Tensorflow implementation of smooth L1 loss defined in Fast RCNN:
        (https://arxiv.org/pdf/1504.08083v2.pdf)

                    0.5 * (sigma * x)^2         if |x| < 1/sigma^2
    smoothL1(x) = {
                    |x| - 0.5/sigma^2           otherwise
    '''
    with tf.variable_scope('smoothL1'):
        conditional = tf.less(tf.abs(x), 1 / sigma ** 2)

        close = 0.5 * (sigma * x) ** 2
        far = tf.abs(x) - 0.5 / sigma ** 2

    return tf.where(conditional, close, far)

def roi_pool(featureMaps, rois, im_dims):
    '''
    Regions of Interest (ROIs) from the Region Proposal Network (RPN) are
    formatted as:
    (image_id, x1, y1, x2, y2)
    Note: Since mini-batches are sampled from a single image, image_id = 0s
    '''
    with tf.variable_scope('roi_pool'):
        # Image that the ROI is taken from (minibatch of 1 means these will all be 0)
        box_ind = tf.cast(rois[:, 0], dtype=tf.int32 )
        # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]
        boxes = rois[:, 1:]
        normalization = tf.cast(tf.stack([im_dims[:, 1], im_dims[:, 0], im_dims[:, 1], im_dims[:, 0]], axis=1),
                                dtype=tf.float32)
        boxes = tf.div(boxes, normalization)
        boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)  # y1, x1, y2, x2

        # ROI pool output size
        crop_size = tf.constant([14, 14])
        # ROI pool
        pooledFeatures = tf.image.crop_and_resize(image=featureMaps, boxes=boxes, box_ind=box_ind, crop_size=crop_size)
        # Max pool to (7x7)
        pooledFeatures = tf.nn.max_pool(pooledFeatures, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return pooledFeatures , boxes , box_ind

def fast_rcnn(top_conv , sample_rois , rois , im_dims,num_classes , phase_train):
    print '###### Fast R-CNN building.... #######'
    print
    with tf.variable_scope('fast_rcnn'):
        FRCNN_DROPOUT_KEEP_RATE = 0.5
        FRCNN_FC_HIDDEN = [ 1024, 1024 ]
        rois = tf.cond(phase_train , lambda : sample_rois  , lambda : rois )
        pooledFeatures, boxes, box_ind = roi_pool(top_conv, rois, im_dims)# roi pooling
        layer = pooledFeatures  # ? 7,7 128 Same Output
        # print layer
        for i in range(len(FRCNN_FC_HIDDEN)):
            layer = affine('fc_{}'.format(i), layer, FRCNN_FC_HIDDEN[i])
            # Dropout
            layer =tf.cond(phase_train, lambda: tf.nn.dropout(layer, keep_prob=FRCNN_DROPOUT_KEEP_RATE), lambda: layer)

        fast_rcnn_cls_logits = affine('cls', layer, num_classes, activation=None)
        fast_rcnn_bbox_logits = affine('targets', layer, num_classes * 4, activation=None)
    return fast_rcnn_cls_logits , fast_rcnn_bbox_logits

def fast_rcnn_cls_loss(fast_rcnn_cls_score, labels):
    '''
    Calculate the fast RCNN classifier loss. Measures how well the fast RCNN is
    able to classify objects from the RPN.

    Standard cross-entropy loss on logits
    '''
    with tf.variable_scope('fast_rcnn_cls_loss'):
        # Cross entropy error
        fast_rcnn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fast_rcnn_cls_score, labels=labels))
    return fast_rcnn_cross_entropy

def fast_rcnn_bbox_loss(fast_rcnn_bbox_pred, bbox_targets, roi_inside_weights, roi_outside_weights):
    '''
    Calculate the fast RCNN bounding box refinement loss. Measures how well
    the fast RCNN is able to refine localization.
    lam/N_reg * sum_i(p_i^* * L_reg(t_i,t_i^*))
    lam: classification vs bbox loss balance parameter
    N_reg: Number of anchor locations (~2500)
    p_i^*: ground truth label for anchor (loss only for positive anchors)
    L_reg: smoothL1 loss
    t_i: Parameterized prediction of bounding box
    t_i^*: Parameterized ground truth of closest bounding box

    TODO: rpn_inside_weights likely deprecated; might consider obliterating
    '''
    with tf.variable_scope('fast_rcnn_bbox_loss'):
        FRCNN_BBOX_LAMBDA =1
        # How far off was the prediction?
        diff = tf.multiply(roi_inside_weights, fast_rcnn_bbox_pred - bbox_targets)
        diff_sL1 = smoothL1(diff, 1.0)
        # Only count loss for positive anchors
        roi_bbox_reg = tf.reduce_mean(tf.reduce_sum(tf.multiply(roi_outside_weights, diff_sL1), reduction_indices=[1]))
        # Constant for weighting bounding box loss with classification loss
        roi_bbox_reg = FRCNN_BBOX_LAMBDA * roi_bbox_reg

    return roi_bbox_reg

def nms_fast_rcnn_blobs(fast_blobs , fast_preds):
    cls_keep={}
    assert len(fast_blobs) == len(fast_preds)
    n_classes  = len(fast_preds[0])
    gt_cls  = np.argmax(fast_preds, axis=1)

    for c in range(0, n_classes):
        target_indices = np.where([gt_cls == c  ])[1]
        target_blobs = fast_blobs[target_indices , c:c+4]
        target_preds = fast_preds[target_indices  , c]
        assert len(target_blobs) == len(target_preds)
        if len(target_blobs) == 0:
            continue;
        target_preds=target_preds.reshape([-1 ,1])
        keep = non_maximum_supression(np.hstack([target_blobs , target_preds]) , 0.7)
        cls_keep[c] = target_indices[keep]
    return cls_keep


def bbox_transform_inv(boxes, deltas):
    '''
    Applies deltas to box coordinates to obtain new boxes, as described by
    deltas
    '''
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes

def get_interest_target_py(cls ,bboxes , n_classes ):
    if np.ndim(bboxes) == 3 :
        bboxes = np.reshape(bboxes , np.shape(bboxes)[1:])
    assert len(cls) == len(bboxes) , 'cls : {} bboxes : {}'.format(np.shape(cls) , np.shape(bboxes))
    assert  np.shape(bboxes)[-1] == n_classes * 4

    #cls = np.argmax(cls , axis =1 )
    ret_targets=[]
    for i in range(len(cls)):
        ret_targets.append(bboxes[ i , cls[i]*4 :(cls[i]+1)*4 ])
    return np.asarray(ret_targets)




def get_interest_target(cls ,bboxes , n_classes):
    target_blobs = tf.py_func(get_interest_target_py , [cls , bboxes , n_classes] , [tf.float32] )
    return target_blobs








