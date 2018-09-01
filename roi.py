import tensorflow as tf
import proposal_layer
def _rpn_softmax(rpn_cls_layer):
    shape = tf.shape(rpn_cls_layer) # 1, h ,w , 18
    rpn_cls_score = tf.transpose(rpn_cls_layer, [0, 3, 1, 2])  # Tensor("transpose:0", shape=(1, 18, h, w)
    rpn_cls_score = tf.reshape(rpn_cls_score, [shape[0], 2, shape[3] // 2 * shape[1],shape[2]]) # shape=(1, 2,h*9, w)
    rpn_cls_score = tf.transpose(rpn_cls_score,[0, 2, 3, 1]) # shape=(?, h*9, w, 2)
    rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])# shape=(h*9*w, 2)
    rpn_cls_prob = tf.nn.softmax(rpn_cls_score) # # shape=(h*9*w, 2)

    # Reshape back to the original
    rpn_cls_prob = tf.reshape(rpn_cls_prob, [1,( shape[3] // 2 )*shape[1], shape[2],  2]) #(1,A*h , w, 2)
    rpn_cls_prob = tf.transpose(rpn_cls_prob, [0,3,1,2]) #(1,2 , h,w)
    rpn_cls_prob = tf.reshape(rpn_cls_prob, [1, 2*(shape[3] // 2), shape[1],  shape[2]]) #(1,18 , h, w)
    rpn_cls_prob = tf.transpose(rpn_cls_prob, [0,2,3,1])  # (1, h, w , 18)



    """
    rpn_cls_prob = tf.transpose(rpn_cls_prob_ori,
                                [0, 3, 1, 2])  # Tensor("transpose_2:0", shape=(?, 2, ?, ?), dtype=float32)
    rpn_cls_prob = tf.reshape(rpn_cls_prob, [shape[0], shape[3], shape[1], shape[
        2]])  # Tensor("transpose_2:0", shape=(?, ?, ?, ?), dtype=float32)
    rpn_cls_prob = tf.transpose(rpn_cls_prob,
                                [0, 2, 3, 1])  # Tensor("transpose_2:0", shape=(?, ?, ?, ?), dtype=float32)
    """
    return rpn_cls_prob


def roi_proposal(rpn_cls_layer , rpn_bbox_layer, im_dims , _feat_stride , anchor_scales , is_training):
    print '########################################################'
    print '########## ROI Proposal Network building.... ###########'
    print '########################################################'
    rpn_cls_prob = _rpn_softmax(rpn_cls_layer)

    blobs, scores , blobs_ori , scores_ori = proposal_layer.proposal_layer(rpn_bbox_cls_prob=rpn_cls_prob,
                                                            rpn_bbox_pred=rpn_bbox_layer,
                                                            im_dims=im_dims, cfg_key=is_training,
                                                            _feat_stride=_feat_stride,
                                                            anchor_scales=anchor_scales)
    blobs = tf.identity(blobs, 'roi_blobs_op')
    scores = tf.identity(scores, 'roi_scores_op')

    return blobs , scores ,blobs_ori , scores_ori ,rpn_cls_prob

