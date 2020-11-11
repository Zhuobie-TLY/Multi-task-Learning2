import numpy as np
import cv2
import keras.backend as K
import tensorflow as tf


a = K.variable(np.array([[[1, 2], [2, 3], [3, 4]]]))
b = K.variable(np.array([[[7, 7], [8, 8], [9, 9]]]))
c1 = K.concatenate([a , b] , axis=0)
c2 = K.concatenate([a , b] , axis=1)
c3 = K.concatenate([a , b] , axis=2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(c1))
    print(sess.run(c2))
    print(sess.run(c3))
# [[1. 2. 3.]
#  [3. 2. 1.]]
# [[1. 2. 3. 3. 2. 1.]]