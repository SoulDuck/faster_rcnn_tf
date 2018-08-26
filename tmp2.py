import numpy as np
array_ = np.asarray([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
a = np.asarray([1,1,1,1])

print array_ - a


def tmp_(*args):
    sums=map(np.sum , args)
    return sums


if __name__ == '__main__':
    print tmp_([1,2,3],[4,5,6])