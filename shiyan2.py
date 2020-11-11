import numpy as np
import tensorflow as tf
from keras.layers import Permute, Dense, TimeDistributed, Conv2D, MaxPooling2D, Multiply, Dropout, Flatten, Reshape, \
    LSTM, UpSampling2D, concatenate, BatchNormalization
from keras import Input, Model
from keras import backend as K
import networkx as nx
import scipy.sparse as sp
from keras.regularizers import l2
from keras.optimizers import Adam

A = tf.constant([[1, 2, 3, 4], [2, 2, 2, 2],[3, 3, 3, 3]])
#,[2, 2, 2, 2],[3, 3, 3, 3]
print(A)
A = tf.reshape(A, [-1, 1])
print(A)
B = tf.reshape(A, [-1, 3, 2])
with tf.Session() as sess:


    print(B.eval())

A = [[1, 1], [2, 2],[3, 3]]#3*2
B = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
W = A+B
print(W)

X = Input(shape = (1433, 1,))
print('jksafhjsdf')

VV = np.array(((1, 2),(1,2)))
print(VV.shape)
W = np.zeros(shape=(4,2,2))
print(W.shape)
Q = W+VV
print(Q)

supplyX_train = np.load("supply&demand_GF.npz")
print(supplyX_train['arr_0'].shape)