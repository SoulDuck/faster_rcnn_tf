import numpy as np
import tensorflow as tf


label=np.asarray([[0,0,1] , [0,1,0] , [1,0,0]] , dtype=np.int32)
bbox = np.asarray([[1,2,3,4,5,6,7,8,9,10,11,12] ,
                   [13,14,15,16,17,18,19,20,21,22,23,24] ,
                   [25,26,27,28,29,30,31,32,33,34,35,36]] , dtype=np.int32)

label_cls = tf.argmax(label , axis=1)

range=tf.range(0,3)
label_op = tf.Variable(label , dtype=tf.int32)
bbox_op = tf.Variable(bbox ,dtype=tf.int32)
counter = tf.Variable(0)
def body(counter):
    result =bbox_op[counter]
    counter = counter + 1

    return result

def condition(counter):
    return counter < 3

with tf.Session():
    tf.initialize_all_variables().run()
    result = tf.while_loop(condition, body, [counter] ,shape_invariants=[12,],back_prop=False)
    #tf.stack(result , axis=0)
    print(result.eval())


