#-*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy.random as npr
import random
import cv2
# ref : https://github.com/mdbloice/Augmentor
class Imgaug(object):
    def __init__(self):
        pass;

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

class TiltImages(Imgaug):
    def __call__(self, np_img, coordinates , angle):
        tform_img = self.tilt_image(np_img, angle)
        tform_coords = self._mask_transform(np_img.shape[:2], coordinates , self.tilt_image , angle)
        return tform_img , tform_coords

class RotationTransform(Imgaug):
    def __call__(self, np_img , coordinates , index):
        # rt = Rotate , Roatate Image
        tform_img=self.rotate90(np_img , index)
        tform_coords = self._mask_transform(np_img.shape[:2], coordinates, self.rotate90, index)
        #masks=self.mappings(coordinates)
        #tform_masks = map(lambda mask: self.rotate90(mask ,index), masks)  # rotated masked
        #tform_coords = self.get_TLBRs(tform_masks )
        return tform_img  , tform_coords

class FlipTransform(Imgaug):
    def __call__(self, np_img , coordinates , index):
        if index > 1 :
            index %= 2

        tform_img = self.flipflop(np_img , index)
        tform_coords = self._mask_transform(np_img.shape[:2], coordinates, self.flipflop, index)
        #masks=self.mappings(coordinates)
        #tform_masks = map(lambda mask: self.flipflop(mask ,index), masks)  # rotated masked
        #tform_coords = self.get_TLBRs(tform_masks )
        return tform_img  , tform_coords

class BrightnessAugmentation(object):
    """
    ref : https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
    ref : https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    """
    def __call__(self , img , k ):
        return self.increase_brightness(img , k )
    def increase_brightness(self , img, k):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - k
        v[v > lim] = 255
        v[v <= lim] += k

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

class ContrastTranform(object):
    """
    ref : https://stackoverflow.com/questions/42045362/change-contrast-of-image-in-pil
    """
    def __call__(self , img , k):
        return self.change_contrast(Image.fromarray(img) , k)
    def change_contrast(self, img, k ):

        factor = (259 * (k + 255)) / (255 * (259 - k))

        def contrast(c):
            return 128 + factor * (c - 128)

        return img.point(contrast)

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
    coords = [coord_bike , coord_car , coord_dog]
    print coords
    k=random.randint(0,10)
    # rotate angle
    imgaug =Imgaug()
    tilt_images = TiltImages()
    t_img , t_coords =tilt_images(np_img , coords , k)
    imgaug.show_image(t_img, t_coords)

    # rotate 90 , 180 ,270
    imgaug =Imgaug()
    rt_images = RotationTransform()
    rt_img , rt_coords = rt_images(np_img , coords , k)
    imgaug.show_image(rt_img, rt_coords)

    # flip flop images
    imgaug = Imgaug()
    ft_images = FlipTransform()
    ft_img, ft_coords = ft_images(np_img, coords, k)
    imgaug.show_image(ft_img, ft_coords)


    """
    img = Image.open('sample_img.png')
    fig = plt.figure()
    ax=fig.add_subplot(111)
    rect = patches.Rectangle((100,100) , 300 ,400)
    ax.add_patch(rect)
    ax.imshow(img)
    plt.close()

    ##
    imgsize = np.shape(img)[:2]
    imgaug=Imgaug()
    mask = imgaug.mapping(imgsize ,(100 ,100 , 400 , 500))
    print imgaug.get_TLBR(mask)
    plt.imshow(mask)
    plt.show()

    ##
    rb_img= imgaug.rotate_bound(np.asarray(img ), 10)
    rb_mask = imgaug.rotate_bound(np.asarray(mask), 10)
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.imshow(rb_mask)
    x1,y1,x2,y2=imgaug.get_TLBR(rb_mask)
    rect = patches.Rectangle((x1,y1),x2-x1, y2-y1)
    ax.add_patch(rect)
    plt.show()

    ##
    rt = RotationTransform()
    rt_mask = rt(mask)
    print imgaug.get_TLBR(rt_mask)
    plt.imshow(rt_mask)
    plt.show()

    ##
    ft = FlipTransform()
    ft_mask = ft(mask)
    print imgaug.get_TLBR(rt_mask)
    plt.imshow(ft_mask)
    plt.show()

    ##
    np_img=np.asarray(img)
    br = BrightnessAugmentation()
    br_img =br(np_img)
    plt.imshow(np_img)
    plt.show()
    plt.close()
    plt.imshow(br_img)
    plt.show()

    ##
    ct = ContrastTranform()
    ct_img=ct(np_img)
    print np.shape(np.asarray(ct_img))
    plt.imshow(ct_img)
    plt.show()

    exit()

"""
