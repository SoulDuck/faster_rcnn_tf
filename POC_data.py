#-*- coding:utf -*-
import glob , os
import argparse
from PIL import Image
import cv2
import numpy as np
from utils import read_gtbboxes
class Poc_datum(object):
    def __init__(self , label_path , images_dir , image_ext = 'jpg'):
        self.root_dir = 'clutteredPOCKIA_TEST_2'
        self.label_path = label_path
        self.img_dirpath = images_dir
        self.img_paths = glob.glob(os.path.join(self.img_dirpath , '*.{}'.format(image_ext)))
        # Get labels
        self.poc_labels_dict = self._arrange_labels()
        #self.show_label_and_images()
        self._count_labels()
        # clutteredPOCKIA folder 및 sub folder 을 생성합니다. 그리고 해당 형식에 맞게 저장합니다
        self.error_indices = self._create_clutteredPOCKIA()
        # 잘못된 좌표값이 있는 이미지는 list 에서 제거 합니다.
        # self.delete_files(self.img_paths , self.error_indices)
        self.img_paths = self.pop_elements(self.img_paths , self.error_indices)
        self.chk_images_labels(os.path.join(self.root_dir , 'Images') ,os.path.join(self.root_dir , 'Annotations') )
        assert len(self.img_paths) == len(self.poc_labels_dict), '# image paths : {} , # labels : {}'.format(
            len(self.img_paths), len(self.poc_labels_dict))

    def _arrange_labels(self):
        # 보기 쉽게 라벨을 dictionary 형태로 정리합니다
        # 라벨이 # image name , n_sample , x1,y1,x2,y2, label name , .... 이런형태로 되어 있어
        # [image name ] = [[x1,y1,w,h,labelname ] , [x1`,y1`,w` , h` , label name ] ..] 이렇게 저장합니다
        f = open(poc_label_path, 'r')
        lines = f.readlines()
        poc_dict = {}
        for line in lines[:]:
            elements = line.split(',')
            elements = map(lambda ele : ele.strip() , elements ) # \r\n 을 제거합니다 .
            image_name, n_samples = elements[:2]
            image_name , ext = os.path.splitext(image_name)
            for i in range(int(n_samples)):
                start = 2 + (5 * (i))
                end = start + 5
                if i ==0 :
                    poc_dict[image_name] = [elements[start : end ]]
                else:
                    poc_dict[image_name].append(elements[start : end ])
        return poc_dict


    def _create_clutteredPOCKIA(self):
        # Annotations , Images , Names folder 을 생성합니다

        anns_path, imgs_path, names_path = map(lambda dirname: os.path.join(self.root_dir, dirname),
                                               ['Annotations', 'Images', 'Names'])
        # create folder
        map(self.chk_dir_and_create , [anns_path , imgs_path , names_path])
        # copy images
        traintxt = open(os.path.join(names_path, 'train.txt'), 'w')
        error_indices = []
        # 이미지가 있는 것들중에 label 이 있는것을 추출한다
        for i,path in enumerate(self.img_paths):
            name = os.path.split(path)[-1]
            name, ext = os.path.splitext(name)
            try:
                labels = self.poc_labels_dict[name]
                labeltxt = open(os.path.join(anns_path, '{}.txt'.format(name)), 'w')
            except KeyError as ke:
                print 'Error name {} '.format(name)
                error_indices.append(i)
                continue


            for label in labels:
                x,y,w,h,lab = label[:]
                x1,y1,x2,y2=self.xywh2xyxy(x,y,w,h)
                lab= lab[-1].replace('label' , '')
                labeltxt.write('{} {} {} {} {}\n'.format(x1,y1,x2,y2,lab))
                traintxt.write('{}\n'.format(name))
            labeltxt.close()
        traintxt.close()
        return error_indices

        # create labels
        # label shape : x1, y1, x2 , y2 , label

        # create name

    def chk_dir_and_create(self , path ):
        # check dir
        if not os.path.isdir(path):
            os.makedirs(path)
        else:
            pass;

    def xywh2xyxy(self , x,y,w,h):
        x,y,w,h=map(int , [x,y,w,h])
        return x, y, x + w, y + h


    # 데이터셋을 Train , Validation , Test set 으로 나눕니다
    def divide_TVT(self , paths , val_ratio , test_ratio):
        n_val = len(paths) * val_ratio
        n_test = len(paths) * test_ratio

    def pop_elements(self , target_list , indices):
        rev_indices=sorted(indices , reverse=True)
        for i in rev_indices:
            target_list.pop(i)
        return target_list

    def delete_files(self , paths , indices):
        for i in indices:
            os.remove(paths[i])

    def chk_images_labels(self , images_dir , labels_dir):
        img_paths=glob.glob(os.path.join(images_dir  , '*.jpg'))
        #lab_paths = glob.glob(os.path.join(labels_dir, '*.txt'))
        err_count = 0
        for img_path in img_paths:
            name = os.path.splitext(os.path.split(img_path)[-1])[0]
            if not os.path.exists(os.path.join(labels_dir , '{}.txt'.format(name))):
                print 'error name : {}'.format(name)
                err_count += 1
        print err_count

    def show_label_and_images(self):
        sample_dir= './clutteredPOCKIA/sample_images'
        lab_1_2_img=os.path.join(sample_dir , '20180109_130752_0090.jpg')
        lab_3_4_img = os.path.join(sample_dir, '20180109_130752_0314.jpg')
        lab_5_6_img = os.path.join(sample_dir, '20180109_130816_0218.jpg')
        lab_7_8_img = os.path.join(sample_dir, '20180109_130752_0564.jpg')
        print self.poc_labels_dict['20180109_130752_0090']
        print self.poc_labels_dict['20180109_130752_0314']
        print self.poc_labels_dict['20180109_130816_0218']
        print self.poc_labels_dict['20180109_130752_0564']

        def _draw_rect( img  , coordinates  ):
            # coordinates = [x1,y1,x2,y2,label]
            x,y,w,h,label = coordinates
            x,y,w,h=map(int , [x,y,w,h])
            img = cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0) , 3)
            img = cv2.putText(img, label ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return img

        img=cv2.imread(lab_1_2_img )
        img = _draw_rect(img, self.poc_labels_dict['20180109_130752_0090'][0])
        img = _draw_rect(img, self.poc_labels_dict['20180109_130752_0090'][1])
        img = cv2.imread(lab_3_4_img)
        img = _draw_rect(img, self.poc_labels_dict['20180109_130752_0314'][0])
        img = _draw_rect(img, self.poc_labels_dict['20180109_130752_0314'][1])
        img = cv2.imread(lab_5_6_img)
        img = _draw_rect(img, self.poc_labels_dict['20180109_130816_0218'][0])
        img = _draw_rect(img, self.poc_labels_dict['20180109_130816_0218'][1])
        img = cv2.imread(lab_7_8_img)
        img = _draw_rect(img, self.poc_labels_dict['20180109_130752_0564'][0])
        img = _draw_rect(img, self.poc_labels_dict['20180109_130752_0564'][1])


    def _count_labels(self):
        labels = {}
        for key in self.poc_labels_dict.keys():
            coords=self.poc_labels_dict[key]
            for coord in coords:
                x,y,w,h,label = coord
                if not label in labels.keys():
                    labels[label] =0
                else:
                    labels[label] += 1

        print labels



def missing_data(ori_dir , label_txt):

    ori_paths = glob.glob(os.path.join(ori_dir, '*.jpg'))
    assert len(set(ori_paths)) == len(ori_paths)
    ori_names = map(lambda path :  os.path.split(path)[-1], ori_paths )


    f= open(label_txt ,'r')
    lines = f.readlines()
    fnames = []
    for line in lines:
        fname=line.split('\t')[0]
        fnames.append(fname)
    fnames = (set(fnames))
    ori_names = set(ori_names)
    return ori_names - fnames


if __name__ == '__main__':

    """
    ori_dir = '/Users/seongjungkim/Desktop/1차리뷰용_데이터/test'
    label_path = '/Users/seongjungkim/PycharmProjects/faster_rcnn_tf/result_test_best_candidates_images/best_labels.txt'
    print missing_data( ori_dir , label_path )
    exit()


    error_list = []
    dict = read_gtbboxes('test_labels.txt')
    for key in dict:
        labels = dict[key]
        tmp_labels = []
        for label in labels:
            tmp_labels.append(label[-1])
        if len(set(tmp_labels)) != len(tmp_labels):
            error_list.append(key)

    print len(error_list)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--images_dir' , type = str)
    args = parser.parse_args()

    poc_label_path = './clutteredPOCKIA_TEST_2/poc_labels.txt' #args.label_path
    poc_images_dir = './clutteredPOCKIA_TEST_2/Images/' #args.images_dir
    poc_datum = Poc_datum(poc_label_path , poc_images_dir)


