# -*- coding:utf-8 -*-
import numpy as np
from utils import non_maximum_supression

def get_recall_precision( scores, groundtruths):
    assert len(scores) == len(groundtruths)
    scores_gts = np.hstack(
        (np.asarray(scores).reshape((-1, 1)), np.asarray(groundtruths).reshape((-1, 1))))  # scr = scores
    order = scores_gts[:, 0].argsort()[::-1]
    scores_gts = scores_gts[order]  # sort the list by ascenging order
    n_gts = len(scores_gts[:, 1] == True)
    n_true_pos = 0.0
    ret_recall = []
    ret_precision = []
    for i, (scr, gt) in enumerate(scores_gts):
        if gt == True:
            n_true_pos += 1
        recall = n_true_pos / n_gts
        precision = n_true_pos / (i + 1)

        ret_recall.append(recall)
        ret_precision.append(precision)
    return ret_recall, ret_precision


def get_iou( pred_bbox, gt_bboxes):
    # 여러개의 gt박스가 있으면 가장 많이 겹치는 gt_bbox이다

    ious = []
    for i,gt_bbox in enumerate(gt_bboxes):

        p_x1, p_y1, p_x2, p_y2 = pred_bbox
        g_x1, g_y1, g_x2, g_y2 = gt_bbox

        xx1 = np.maximum(p_x1, g_x1)
        yy1 = np.maximum(p_y1, g_y1)
        xx2 = np.minimum(p_x2, g_x2)
        yy2 = np.minimum(p_y2, g_y2)

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap_area = w * h
        pred_area = (p_x2 - p_x1 + 1) * (p_y2 - p_y1 + 1)
        gt_area = (g_x2 - g_x1 + 1) * (g_y2 - g_y1 + 1)

        iou = overlap_area / float(pred_area + gt_area - overlap_area)
        ious.append(iou)
    return ious

def get_max_iou(pred_bbox, gt_bboxes):
    return np.max(get_iou(pred_bbox, gt_bboxes))


def get_interpolated_precision(recall , precision , thres_recall):
    #print recall_precision
    #recall = recall_precision[:, 0]
    #precision = recall_precision[:, 1]
    recall , precision =map(np.asarray , [recall , precision])
    indices=[recall >= thres_recall]
    indices=np.where(indices)[1]
    selected_positive=precision[indices]
    if len(selected_positive)==0:
        return 0
    return np.max(selected_positive)


def get_AP(recall,precision): # AP = Average Precision
    precisions=[]
    for i in range(11):
        thres_recall=i/10.
        precisions.append(get_interpolated_precision(recall, precision, thres_recall))
    return np.mean(precisions)


def mean_mAP(*args):
    return np.mean(args)

def get_acc(pred_cls , true_cls):
    pred_cls , true_cls =map(np.asarray , [pred_cls , true_cls])
    assert len(pred_cls) == len(true_cls) and np.ndim(pred_cls) == np.ndim(true_cls) == 1
    return np.sum([pred_cls == true_cls]) / float(len(pred_cls))

def poc_acc(itr_fr_blobs, fr_cls ,gt_bboxes, threshold ):
    # 좀 복잡해 한글로 설명을 길게 남긴다
    pred_ious = []
    gt_classes =gt_bboxes[:, -1:]
    # IOU 가 겹치는 rectangle 가져온다.
    for blob in itr_fr_blobs:
        # get_iou ==> [0.2  , 0.1 , 0.98] , IOUs for each gt bboxes
        pred_ious.append(get_iou(blob , gt_bboxes[:,:-1]))
    pred_ious = np.asarray(pred_ious)
    #
    ious_argmax = np.argmax(pred_ious , axis =1 )
    # gt box 순서와 최고 iou 가있는걸 가져온다 .
    # 해당 gt bboxes 와 가장 많이 겹치는 blobs index 을 가져온다 ==> 0.5 이상으로 겹치는 걸 확인한다
    # 가장 ious 가 높은 것을 추출하고 그중 threshold 보다 높은 iou을 가진 indices 을 가져온다
    ious_max = np.max(pred_ious , axis =1 )
    over_thr_indices =np.where([ious_max >threshold])[1]
    for i, gt_bbox in enumerate(gt_bboxes):
        gt_cls = gt_classes[i]
        indices = np.where([ious_argmax  == i])[1]
        ## iou 가 threshold 보다  overlay indices 가 해당 gt 인 교집합을 찾는다 그러면 겹치는 게 나온다
        assert len(set(over_thr_indices)) == len(over_thr_indices) and len(set(indices)) == len(indices)
        target_indices = set(over_thr_indices) & set(indices)
        target_pred_cls = fr_cls[list(target_indices)]
        target_pred_cls = map(int ,target_pred_cls)
        if not len(target_pred_cls) ==0 :
            gt_cls = gt_cls * len(target_pred_cls)
            target_acc = np.sum([gt_cls ==target_pred_cls]) / float(len(target_pred_cls))
        else:
            target_acc = 0
        print '{} , accuracy : {}'.format(i, target_acc)
        print 'fast rcnn cls : '.format(target_pred_cls)

if __name__ == '__main__':
    ious = [0.78, 0.88, 0.76, 0.43, 0.44]
    pred_bbox = [0, 0, 10, 10]
    gt_bboxes = [[5, 5, 15, 15], [7, 7, 15, 15]]
    print get_iou(pred_bbox, gt_bboxes)

    scores = [0.97, 0.96, 0.94, 0.93, 0.91, 0.89, 0.78, 0.77, 0.76, 0.56]
    gt = [True, True, True, False, True, True, False, True, True, True]
    recall, precision = get_recall_precision(scores, gt)
    ap = get_AP(recall, precision)
    print get_acc([1,2,3] , [1,2,6])