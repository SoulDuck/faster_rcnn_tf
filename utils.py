#-*- coding:utf-8 -*_
import os
from scipy.misc import imread
from PIL import  Image
import numpy as np
import matplotlib
import tensorflow as tf
import sys , os
from PIL import Image
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import glob
import numpy as np
def progress(i, max_step):
    msg = '\r {} / {}'.format(i, max_step)
    sys.stdout.write(msg)
    sys.stdout.flush()
def next_img_gtboxes(image_idx):
    IMAGE_FORMAT= '.jpg'
    data_dir='./clutteredPOCKIA_TRAIN'
    train_name_path = os.path.join(data_dir, 'Names', 'train.txt')
    train_names = [line.rstrip() for line in open(train_name_path, 'r')]
    if image_idx > (len(train_names)-1) :
        image_idx= image_idx % (len(train_names)-1)


    img_path = os.path.join(data_dir, 'Images', train_names[image_idx] + IMAGE_FORMAT)
    annotation_path = os.path.join(data_dir, 'Annotations', train_names[image_idx] + '.txt')
    img = Image.open(img_path).convert('RGB')
    img=np.asarray(img)
    #img = imread(img_path)

    gt_bbox = np.loadtxt(annotation_path, ndmin=2)
    #im_dims = np.array(img.shape[:2]).reshape([1, 2])

    flips = [0, 0]
    flips[0] = np.random.binomial(1, 0.5)
    #img = image_preprocessing.image_preprocessing(img)
    if np.max(img) > 1:
        img = img / 255.
    return img , gt_bbox


def next_img_gtboxes_with_path(image_idx):
    IMAGE_FORMAT= '.jpg'
    data_dir='./clutteredPOCKIA_TEST'
    train_name_path = os.path.join(data_dir, 'Names', 'train.txt')
    train_names = [line.rstrip() for line in open(train_name_path, 'r')]
    if image_idx > (len(train_names)-1) :
        image_idx= image_idx % (len(train_names)-1)


    img_path = os.path.join(data_dir, 'Images', train_names[image_idx] + IMAGE_FORMAT)
    annotation_path = os.path.join(data_dir, 'Annotations', train_names[image_idx] + '.txt')
    img = Image.open(img_path).convert('RGB')
    img=np.asarray(img)
    #img = imread(img_path)

    gt_bbox = np.loadtxt(annotation_path, ndmin=2)
    #im_dims = np.array(img.shape[:2]).reshape([1, 2])

    flips = [0, 0]
    flips[0] = np.random.binomial(1, 0.5)
    #img = image_preprocessing.image_preprocessing(img)
    if np.max(img) > 1:
        img = img / 255.
    return img , gt_bbox , img_path

def non_maximum_supression(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]]) # xx1 shape : [19,]
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h #inter shape : [ 19,]
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def draw_rectangles_fastrcnn(img , bboxes , true_classes  , savepath):
    ax = plt.axes()
    bboxes = np.asarray(bboxes)
    bboxes = np.squeeze(bboxes)
    plt.imshow(img)
    h,w = np.shape(img)[:2]
    pos_indices=np.where([true_classes > 0])[1]
    neg_indices = np.where([true_classes == 0])[1]
    pos_bboxes = bboxes[pos_indices]
    neg_bboxes = bboxes[neg_indices]

    for box in pos_bboxes :
        x1, y1, x2, y2= box[:]  # x1 ,y1 ,x2 ,y2
        if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        else:
            continue

    for box in neg_bboxes :
        x1, y1, x2, y2= box  # x1 ,y1 ,x2 ,y2
        if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        else:
            continue

    plt.savefig(savepath)
    plt.close()



def draw_rectangles_ptl(img,ptl_bbox , inv_fast_bbox , ptl_labels  , inv_fast_cls ,  cls_keep , savepath):
    bg_indices = np.where([ptl_labels == 0])[-1]
    fg_indices = np.where([ptl_labels != 0])[-1]
    bg_bboxes = ptl_bbox[bg_indices]
    fg_bboxes = ptl_bbox[fg_indices]
    h, w = np.shape(img)[:2]
    fig=plt.figure(figsize=[20,20])
    ax=fig.add_subplot(111)
    plt.imshow(img)
    for fg_ind , box in enumerate(fg_bboxes[:,1:]):
        x1, y1, x2, y2 = box  # x1 ,y1 ,x2 ,y2
        plt.text(x1,y1,'{}'.format(ptl_labels[fg_indices[fg_ind]]))
        if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        else:
            continue
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(savepath.replace('roi' , 'ptl_fg_roi'), bbox_inches=extent)
    #plt.savefig(savepath.replace('roi' , 'ptl_fg_roi'))
    plt.close()

    fig=plt.figure(figsize=[20,20])
    ax=fig.add_subplot(111)
    plt.imshow(img)
    for box in bg_bboxes[:,1:] :
        x1, y1, x2, y2= box  # x1 ,y1 ,x2 ,y2
        if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        else:
            continue
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(savepath.replace('roi' , 'ptl_bg_roi'), bbox_inches=extent)
    #plt.savefig(savepath.replace('roi' , 'ptl_bg_roi'))
    plt.close()

    fig=plt.figure(figsize=[20,20])
    ax=fig.add_subplot(111)
    plt.imshow(img)
    for i,box in enumerate(inv_fast_bbox[:len(fg_indices)]):
        pred = inv_fast_cls[i]
        true = ptl_labels[i]
        start = pred*4
        end= (pred+1) * 4
        x1, y1, x2, y2= box[start : end ]
        plt.text(x1,y1,str('{}:{}'.format(pred , true)) , fontsize=12 ,color='red')
        if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        else:
            continue

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(savepath.replace('roi', 'ptl_inv_roi'), bbox_inches=extent)
    #plt.savefig(savepath.replace('roi' , 'ptl_inv_roi'))
    plt.close()


    ax = plt.axes()
    plt.imshow(img)
    for c in cls_keep:
        if c == 0:
            continue
        for i in cls_keep[c]:
            box = inv_fast_bbox[i]
            pred = inv_fast_cls[i]
            true= ptl_labels[i]
            start = pred * 4
            end= (pred+1) * 4

            x1, y1, x2, y2= box[start : end ]
            plt.text(x1,y1,str('{}:{}'.format(pred , true)) , fontsize=12 ,color='red')
            if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
            else:
                continue
    plt.savefig(savepath.replace('roi' , 'ptl_inv_nms_roi'))
    plt.close()


def draw_rectangles(img ,bboxes ,scores , anchors, fname):
    img=np.squeeze(img)
    h,w=np.shape(img)[:2]
    # get Positive indices
    pos_bboxes_indices = np.where([scores >= 0.5])[1]
    pos_bboxes=bboxes[pos_bboxes_indices]
    pos_bboxes = pos_bboxes[:,1:]
    # get Negative indices
    neg_bboxes_indices = np.where([scores < 0.5])[1]
    neg_bboxes = bboxes[neg_bboxes_indices]
    neg_bboxes = neg_bboxes[:,1:]

    # makedirs
    def _makedir(dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    #
    def _draw_bboxes(img , bboxes ,savepath):
        ax = plt.axes()
        plt.imshow(img)
        for box in bboxes:
            x1, y1, x2, y2 = box  # x1 ,y1 ,x2 ,y2
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
                # add rectangle Patch
                ax.add_patch(rect)
            else:
                continue
        # save figure
        plt.savefig(savepath)
        plt.close()

    # Make dirs
    map(lambda dirpath: _makedir(dirpath),['result_pos_roi', 'result_neg_roi', 'reuslt_anchors'])
    # Join dirpath to filename
    pos_fpath, neg_fpath, anc_fpath = map(lambda dir: os.path.join(dir, fname),
                                          ['result_pos_roi', 'result_neg_roi', 'reuslt_anchors'])
    # DRAW POS BBOX
    _draw_bboxes(img,pos_bboxes , savepath = pos_fpath)
    _draw_bboxes(img,neg_bboxes , savepath = neg_fpath)
    # anchor shape : x1 y1 x2 y2 label
    _draw_bboxes(img,anchors[:,:4], savepath= anc_fpath)


def tracking_rpn_loss(summary_writer, loss, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag='rpn_loss', simple_value=float(loss))])
    summary_writer.add_summary(summary, step)


def tracking_loss(summary_writer, loss, step, prefix ):
    summary = tf.Summary(value=[tf.Summary.Value(tag='{}'.format(prefix), simple_value=float(loss))])
    summary_writer.add_summary(summary, step)


def read_image(path):
    img = np.asarray(Image.open(path).convert('RGB'))
    h,w,ch= np.shape(img)
    return np.reshape(img , [1,h,w,ch])
def read_gtbboxes(path):
    f=open(path,'r')
    lines=f.readlines()
    gt_boxes= []
    for line in lines:
        x1,y1,x2,y2,label=line.split(' ')
        x1,y1,x2,y2=map(float , [x1,y1,x2,y2])
        label =int(label.strip())
        gt_boxes.append([x1,y1,x2,y2,label])
    gt_boxes=np.asarray(gt_boxes)
    return gt_boxes

def get_name(path):
    name = os.path.split(path)[-1]
    name , ext =os.path.splitext(name)
    return name

def draw_fr_bboxes(img , fastrcnn_cls , fastrcnn_bboxes , color , linewidth  , savepath):
    img=np.squeeze(img)
    img=np.asarray(img)*255
    for i, bbox in enumerate(fastrcnn_bboxes):

        cls = fastrcnn_cls[i][0]
        if cls ==0 :
            continue;
        font = cv2.FONT_HERSHEY_SIMPLEX
        x1,y1,x2,y2 = bbox
        cv2.putText(img , '{}'.format(cls) ,(x1,y1) ,font ,1 ,color)
        cv2.rectangle(img , (x1,y1) , (x2,y2) , color ,linewidth )
    cv2.imwrite(savepath, img)


def best_rect(cls , scores , bboxes):
    assert len(cls) == len(scores) == len(bboxes)
    # 이 메스드는 반드시 버튼이 동시에 여러개 없다는 가정이 있어야 한다는 조건이 있다
    best_scores = []
    best_bboxes = []
    best_cls = []
    fg_dict = {} # key : classs values : scores
    for i,c in enumerate(cls):
        if c ==0 :
            continue
        else:
            if not c in fg_dict.keys():
                fg_dict[c] = [i,scores[i] , bboxes[i]]
            else:
                old_score = fg_dict[c][1]
                new_score = scores[i]
                if old_score < new_score:
                    print 'old score : {} new score : {}'.format(old_score , new_score)
                    fg_dict[c] = [i,scores[i] , bboxes[i]]

    print 'N classes : {} '.format(len(fg_dict))
    for cls in fg_dict.keys():
        best_cls.append(cls)
        i,score , bboxes = fg_dict[cls]

        best_bboxes.append([bboxes])
        best_scores.append(score)
    assert len(best_cls) == len(best_scores) == len(best_bboxes) , '{} {} {}'\
        .format(len(best_cls) ,len(best_scores) ,len(best_bboxes))
    best_cls, best_scores, best_bboxes = map(np.asarray,[best_cls , best_scores , best_bboxes])
    best_bboxes=np.reshape(best_bboxes , [-1,4])
    return best_cls , best_scores , best_bboxes






def read_gtbboxes(label_path):
    # poc_label.txt 을 dict 형태로 return 한다 .

    f=open(label_path  , 'r')
    lines=f.readlines()
    ret_gtbboxes={}
    for l in lines:
        tmp_list = []
        elements = l.split(',')
        img_name = elements[0]
        n_labels = elements[1]

        for n in range(int(n_labels)):
            x,y,w,h ,btn = elements[2+n*5 :2+(n+1)*5 ]
            btn=btn.replace('button', '')
            x, y, w, h, btn = map(int, [x, y, w, h, btn])
            tmp_list.append([x, y, w+x, h+y, btn])
        ret_gtbboxes[img_name] = np.asarray(tmp_list)
    assert len(ret_gtbboxes) == len(lines)
    return ret_gtbboxes

if '__main__' == __name__:
    img , gt_boxes =next_img_gtboxes(image_idx=1)
    ax=plt.axes()
    for box in gt_boxes:
        x1, y1, x2, y2, label = box  # x1 ,y1 ,x2 ,y2
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.imshow(img)
    plt.show()
