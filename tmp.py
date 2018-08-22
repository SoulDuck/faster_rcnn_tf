import numpy as np


def bbox_transform(ex_rois, gt_rois):
    '''
    Receives two sets of bounding boxes, denoted by two opposite corners
    (x1,y1,x2,y2), and returns the target deltas that Faster R-CNN should aim
    for.
    '''
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


a = np.asarray([[1,1,1,1],[2,2,2,2] , [3,3,3,3],[4,4,4,4]])
b = np.asarray([[1,1,1,1] ])
print bbox_transform(a,b)

print a-b

n = len(a.reshape([-1]))
a = range(n)
h = 2
w = 3
ch = 8
a = np.reshape(a,[1,h,w,ch])
print a
a = a.transpose([0,3,1,2])
a = np.reshape(a ,[ 1, 2 , h*4, w ])
a = a.transpose([0,2,3,1])
a=a.reshape([-1,2])


a=a.reshape([1,h*4,w,2])
a=a.transpose([0,3,1,2])
a=a.reshape([1,8,h,w])
a=a.transpose([0,2,3,1])

print a
print np.shape(a)




