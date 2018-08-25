from utils import get_name
import glob , os

class Wallyprovider(object):

    def __init__(self,img_dir, anno_fpath):
        print 'a'
        paths = glob.glob(os.path.join(img_dir, '*.jpg'))
        self.names = map(lambda path: get_name(path), paths)
        self.anno_savdir = 'clutteredWALLY/Annotations'
        self.anno_fpath = anno_fpath
        self.name_savedir = 'clutteredWALLY/Names'
        self.img_dir = img_dir
        self._make_annos()
        self._make_namees()



    def _make_annos(self):
        # x1,y1,x2,y2,cls
        scr_anno =open(self.anno_fpath , 'r')

        lines=scr_anno.readlines()
        for line in lines[1:]:
            fname ,w, h, cls , x1,y1,x2,y2=line.split(',')
            name=os.path.splitext(fname)[0]
            f = open(os.path.join(self.anno_savdir , name+'.txt' ), 'w')
            elements = [x1,y1,x2,y2]
            elements=map(lambda ele: ele.strip() , elements)
            if 'waldo' in cls:
                x1,y1,x2,y2=elements
                f.write('{} {} {} {} {}'.format(x1,y1,x2,y2,1))
            f.close()

    def _make_namees(self):
        # Make dir
        if not os.path.exists(self.name_savedir):
            os.makedirs(self.name_savedir)
        name_fpath = os.path.join(self.name_savedir , 'train.txt')
        f = open(name_fpath ,'w')
        for name in self.names:
            f.write(name+'\n')
        f.close()











if __name__ == '__main__':
    anno_path = '/Users/seongjungkim/HereIsWally/annotations/annotations.csv'
    img_dir = '/Users/seongjungkim/HereIsWally/images'
    wallyprovider = Wallyprovider(img_dir, anno_path)






