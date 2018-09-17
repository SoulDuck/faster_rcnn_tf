#-*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy.random as npr
import random
import cv2
import math
import copy
import tensorflow as tf
# ref : https://github.com/mdbloice/Augmentor
class Imgaug(object):
    def __init__(self):
        pass;

    def colorAug(self , images , phase_train):
        def _training(image):
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
            return image

        def _eval(image):
            return image
        images = tf.map_fn(lambda image: tf.cond(phase_train, lambda: _training(image), lambda: _eval(image)),
                           images)
        return images


    def mapping(self,img_size , coordinate):
        """
        map coordinate(x1,y1,x2,y2) to original image size
        :param img_size: (image height , image width)
        :return: numpy
        """
        assert len(coordinate) == 4 and np.ndim(coordinate) == 1 and len(img_size) == 2 and np.ndim(img_size) == 1
        x1, y1, x2, y2 = coordinate
        ret_np = np.zeros(img_size)
        ret_np[y1:y2, x1:x2] = 255

        return ret_np
    def mappings(self , img_size ,coordinates):
        return map(lambda coord : self.mapping(img_size , coord) , coordinates)
    def get_TLBR(self , np_img):
        """
        Top Left , Buttom Right  ==> TLRB
        get Top Left , Buttom Right Coordinate from Region of interested
        :param np_img:
        :return:
        """
        h_axis_indices = np.where([np.mean(np_img, axis=1) > 0])[1]
        w_axis_indices = np.where([np.mean(np_img, axis=0) > 0])[1]

        h_start = h_axis_indices[0]
        h_end = h_axis_indices[-1]
        w_start = w_axis_indices[0]
        w_end = w_axis_indices[-1]

        return  w_start  , h_start , w_end , h_end # x1,y1,x2,y2

    def get_TLBRs(self , np_imgs):
        return map(lambda img: self.get_TLBR(img) , np_imgs)

    def tilt_image(self , image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def rotate90(self , np_img , index ):
        """
        rotate numpy 2d image by Index
        :param np_img:
        :return: np_img
        """
        np_img = np.rot90(np_img, index )
        return np_img
    def flipflop(self,np_img, index ):
        """
        # 0 = Nothing
        # 1 = left right flip flop
        # 2 = up doen flip flop

        randomly flip flop numpy 2d image
        :param np_img:
        :return: np_img
        """
        np_img = np.flip(np_img, index)
        return np_img

    # Callback Function
    def _mask_transform(self ,img_size ,coordinates , func , index ):
        masks=self.mappings(img_size, coordinates)
        # transformed ==> tf
        tf_masks = map(lambda mask: func(mask , index), masks)  # rotated masked
        tf_coords = self.get_TLBRs(tf_masks)
        return tf_coords

    def show_image(self , image ,coordinates , prefix=None):
        """
        Show image with a couple of coordinates
        :param image:
        :param coordinates:
        :return:
        """
        fig =plt.figure()
        ax=fig.add_subplot(111)
        ax.imshow(image)
        ax.set_xlabel(prefix)
        # patch rectangles to image
        for coord in coordinates:
            x1,y1,x2,y2 = coord
            rect = patches.Rectangle((x1,y1) ,x2-x1 , y2- y1 , fill = '')
            ax.add_patch(rect)
        plt.show()
    def limit_constant(self , value , limit_range ):
        """
        :param value:
        :param limit: if param `limit_range` is 10 , it restrict param `value` from -10 to 10
        :return:
        """
        assert limit_range > 0
        if value < 0 :
            value = abs(value) % limit_range
            value = -value
            print value
        else:
            value = value % limit_range
        return value
    def choice_var(self , range_ = range(-10 , 10)): #
        """
        choose Random Variable at input list
        :param range_:
            type : list
            example : [-10, -9 , -8 , 7 ]
        :return: int value
        """
        assert type(range_) == list
        random.shuffle(range_)
        return range_[0]

    def get_img_anns(self , sample_dict):
        assert type(sample_dict) == dict and 'img' in sample_dict.keys() and 'anns' in sample_dict.keys()
        np_img=np.asarray(sample_dict['img'])
        anns = sample_dict['anns']
        return np_img , anns
    def set_img_anns(self, sample_dict , np_img , anns ):
        assert type(sample_dict) == dict
        sample_dict['img'] = Image.fromarray(np_img)
        sample_dict['anns'] = anns
        return sample_dict

    def get_minmax(self , np_img , anns ):
        """
        get min x1 ,min y1 ,max x2 ,y2 from multiple coordis
        example [[x1,y1,w,h],[x1`,y1`,w`,h`]]

        :param np_img:
        :param anns:
        :return:
        """
        assert len(np.shape(np_img)) == 3 and np.ndim(np.asarray(anns)) == 2, 'ndim : {} , shape {}'.format(
            np.ndim(np.asarray(anns)), np.shape(np_img))
        x1s , y1s , x2s , y2s =map(lambda ind : np.asarray(anns)[:,ind] , [0,1,2,3])
        min_x1 , min_y1= map(min , [x1s , y1s])
        max_x2, max_y2 = map(max , [x2s, y2s])
        return min_x1 , min_y1 , max_x2 , max_y2


    def _cal_margin(self , img , anns):
        assert np.ndim(img) ==2 or np.ndim(img) ==3
        min_x1, min_y1, max_x2, max_y2 = self.get_minmax(img, anns)
        img_h,img_w=np.shape(img)[:2]
        assert img_h > max_y2 and img_w > max_x2

        LT_range=[(0,min_x1),(0,min_y1)]
        RT_range = [( max_x2, img_w),(0 , min_y1)]
        RB_range=[(max_x2 , img_w),(max_y2 ,img_h )]
        LB_range = [(0 , min_x1),(max_y2 , img_h)]

        return LT_range , LB_range , RB_range , RT_range

    def get_L2(self , coordi):

        """
        get l2 distance from crop x1 , crop y1 , crop x2 , crop y2
        :param np_img:
        :param coordi: [(x1,x2),(y1,y2)]
        :return:
        """
        xs , ys =coordi
        x1, x2 = xs
        y1 ,y2 = ys
        dist = math.sqrt((x2- x1 )**2 + (y2 - y1) **2)
        return  dist

    def get_L2s (self , *args):
        """
        :param args: LT , LB , RB , RT
        :return:
        """
        return map(self.get_L2 , args)

    def get_ctr(self , coordi):
        """
        get center point  from   x1 ,y1 , x2 , y2
        :param np_img:
        :param coordi: [x1,y1,x2,y2]
        :return:
        """
        x1,y1,x2,y2=coordi
        x_ctr = ((x2 - x1) /2)
        y_ctr = ((y2 - y1) / 2)
        return x_ctr, y_ctr










class TiltImages(Imgaug):
    def __init__(self , angles):
        self.angles = angles
    def __call__(self, sample_dict):
        """
        sample_dict:
            key : 'img' , 'anns'
            value : 'img' = numpy image , 'anns' : list [[x1,y1,w,h] ,[x1`,y1`,w`,h`] ...]

        term :
            tform ==> transformed
        :return:
        """
        # choice one from self.angles
        angle = random.choice(self.angles)

        sample_dict = copy.deepcopy(sample_dict)
        # get image , coordinates
        np_img , coordinates = self.get_img_anns(sample_dict)
        # select one variable at list
        # angle = self.choice_var(range(-10 , 10))
        # transformed Numpy image
        tform_img = self.tilt_image(np_img, angle)
        # transformed coorinates  ,[[x1,y1,w h] ,[x1`,y1`,w`,h`]]
        tform_coords = self._mask_transform(np_img.shape[:2], coordinates , self.tilt_image , angle)
        # set sample_dict
        new_sample_dict=self.set_img_anns(sample_dict , tform_img , tform_coords)
        return new_sample_dict

class RotationTransform(Imgaug):
    def __call__(self, sample_dict):
        sample_dict = copy.deepcopy(sample_dict)
        # get image , coordinates
        np_img , coordinates = self.get_img_anns(sample_dict)
        # select one variable at list
        index = abs(self.choice_var(range(-10 , 10)))
        # rt = Rotate , Roatate Image
        tform_img=self.rotate90(np_img , index)
        tform_coords = self._mask_transform(np_img.shape[:2], coordinates, self.rotate90, index)
        # set sample_dict
        sample_dict=self.set_img_anns(sample_dict , tform_img , tform_coords)


        return sample_dict

class FlipTransform(Imgaug):
    def __call__(self,sample_dict):
        # get image , coordinates
        sample_dict = copy.deepcopy(sample_dict)
        np_img , coordinates = self.get_img_anns(sample_dict)
        # select one variable at list
        # numpy flip flop lib only support value > 0
        index = abs(self.choice_var(range(-10 , 10)))
        if index > 1:
            index %= 2
        tform_img = self.flipflop(np_img , index)
        tform_coords = self._mask_transform(np_img.shape[:2], coordinates, self.flipflop, index)
        new_sample_dict = self.set_img_anns(sample_dict, tform_img, tform_coords)
        return new_sample_dict

class CenterCrop(Imgaug):
    def __call__(self, sample_dict):
        sample_dict = copy.deepcopy(sample_dict)
        # coordi x1, y1, x2, y2
        np_img, coordi = self.get_img_anns(sample_dict)
        h,w=np.shape(np_img)[:2]
        # get x,y center
        x_ctr, y_ctr=self.get_ctr([0,0,w,h])
        # Left top range , shape : ([0,minx1] , [0,min y1]) # Right Bottom , shape : ([max_x2 img_w],[max_y2 ,img_h ])
        LT_range, LB_range, RB_range, RT_range = self._cal_margin(np_img , coordi)
        # Get Max , Min x1,x2 ,y1,y2
        #start_w, min_x1 , start_h , min_y1 =LT_range[0][0], LT_range[0][1] ,LT_range[1][0], LT_range[1][1]
        #end_w, max_x2, end_h, max_y2 = RB_range[0][0], RB_range[0][1], RB_range[1][0], RB_range[1][1]
        # get losses
        lt_l2 , lb_l2 , rb_l2 , rt_l2 = self.get_L2s(LT_range , LB_range , RB_range , RT_range)
        ranges_losses  = zip([LT_range , LB_range , RB_range , RT_range] , [lt_l2 , lb_l2 , rb_l2 , rt_l2])
        #
        ind = np.argmin([lt_l2 , lb_l2 , rb_l2 , rt_l2])
        valid_range , valid_lossses = ranges_losses[ind]
        #
        x_rand = self.choice_var(range(valid_range[0][0] ,valid_range[0][1]))
        y_rand = self.choice_var(range(valid_range[1][0] , valid_range[1][1]))
        #
        x_crop = abs(x_ctr - x_rand)
        y_crop = abs(y_ctr - y_rand)
        x_crop_min = x_ctr - x_crop
        y_crop_min = y_ctr - y_crop
        #
        cropped_img = np_img[y_ctr - y_crop : y_ctr + y_crop, x_ctr - x_crop : x_ctr + x_crop]
        # Get coordinates
        cropped_coordis = coordi - np.asarray([x_crop_min, y_crop_min, x_crop_min, y_crop_min])
        new_sample_dict=self.set_img_anns(sample_dict,cropped_img , cropped_coordis)
        return new_sample_dict


class BrightnessAugmentation(Imgaug):
    """
    ref : https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
    ref : https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    """
    def __init__(self , bright_range):
        self.bright_range = bright_range
        self.k = random.choice(bright_range)
    def __call__(self , sample_dict):
        sample_dict = copy.deepcopy(sample_dict)
        np_img, coordinates = self.get_img_anns(sample_dict)
        np_img = self.increase_brightness(np_img , self.k )
        new_sample_dict = self.set_img_anns(sample_dict, np_img ,coordinates)
        return new_sample_dict



    def increase_brightness(self , img, k):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - k
        v[v > lim] = 255
        v[v <= lim] += k

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return np.asarray(img)

class ContrastTranform(Imgaug):
    """
    ref : https://stackoverflow.com/questions/42045362/change-contrast-of-image-in-pil
    """
    def __init__(self  , k_range):
        self.k = random.choice(k_range)
    def __call__(self , sample_dict):
        np_img, coordinates = self.get_img_anns(sample_dict)
        np_img = self.change_contrast(Image.fromarray(np_img) , self.k)
        new_sample_dict = self.set_img_anns(sample_dict, np_img , coordinates )
        return new_sample_dict


    def change_contrast(self, img, k ):

        factor = (259 * (k + 255)) / (255 * (259 - k))

        def contrast(c):
            return 128 + factor * (c - 128)

        return np.asarray(img.point(contrast))

# ref : https://github.com/ayooshkathuria/pytorch-yolo-v3
# ref : https://github.com/qqwweee/keras-yolo3/blob/master/kmeans.py
"""
# ref : https://medium.com/@vivek.yadav/part-1-generating-anchor-boxes-for-yolo-like-network-for-vehicle-detection-using-kitti-dataset-b2fe033e5807
 in YOLOv2, intersection over union (IOU) is used as a distance metric. The IOU calculations are made assuming all \
 the bounding boxes are located at one point, i.e. only width and height are used as features
"""
class Kmeans(object):
    def __call__(self ,boxes , k , dist ):
        self.kmeans()
    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters
if __name__ == '__main__':
    img = Image.open('dog.jpg').convert('RGB')
    np_img = np.asarray(img)
    coord_bike = [124,134,124+ 443,134+286]
    coord_car = [468,74,468+218,74+96]
    coord_dog = [131,219,131+180,219+324]
    coords = [coord_bike  , coord_car , coord_dog]

    # load Images
    sample_dict = {}
    sample_dict['img'] = np_img
    sample_dict['anns'] = coords
    sample_dict['names'] = 'namees'


    # Tilt
    imgaug=Imgaug()
    tilt_images = TiltImages(range(-10 , 10) + [90,180,270])
    sample_dict =tilt_images(sample_dict )
    t_img = sample_dict['img']
    t_coords = sample_dict['anns']
    imgaug.show_image(t_img, t_coords , 'tilt')

    # rotate 90 ,180 ,270
    rt_images = RotationTransform()
    sample_dict = rt_images(sample_dict)
    rt_img = sample_dict['img']
    rt_coords = sample_dict['anns']
    imgaug.show_image(rt_img, rt_coords , 'rotate')

    # flip flop images
    flip_transform = FlipTransform()
    sample_dict = flip_transform(sample_dict)
    ft_img = sample_dict['img']
    ft_coords = sample_dict['anns']
    imgaug.show_image(ft_img, ft_coords , 'flip flop')

    #center crop
    center_crop= CenterCrop()
    sample_dict=center_crop(sample_dict)
    cc_img = sample_dict['img']
    cc_coords = sample_dict['anns']
    imgaug.show_image(cc_img, cc_coords , 'center crop')

    #BrightnessAugmentation
    bright_aug= BrightnessAugmentation(range(10,100))
    sample_dict = bright_aug(sample_dict )
    ba_img = sample_dict['img']
    ba_coords = sample_dict['anns']
    imgaug.show_image(ba_img, ba_coords , 'bright augmentation')

    #ContrastTranform
    contrast_tranform=ContrastTranform(range(100,200))
    sample_dict = contrast_tranform(sample_dict)
    ct_img = sample_dict['img']
    ct_coords = sample_dict['anns']
    imgaug.show_image(ct_img, ct_coords, 'contrast tranform')