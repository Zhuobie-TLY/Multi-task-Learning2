#!/usr/bin/env Python
#coding=utf-8
import numpy as np
import keras
import tensorflow as tf
from keras.layers import Permute, Dense, TimeDistributed, Conv2D, MaxPooling2D, Multiply, Dropout, Flatten, Reshape, \
    LSTM, UpSampling2D, concatenate, BatchNormalization, Lambda
from keras import Input, Model
from keras import backend as K
import networkx as nx
import scipy.sparse as sp
from keras.regularizers import l2
from keras.optimizers import Adam
from graph import *
from graph8 import GraphConvolution
from utils8 import *
# from keras.utils import plot_model
tf.compat.v1.disable_eager_execution()
#改变matplotlib的默认backend
import matplotlib

from matplotlib import pyplot as plt

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)#实现转置**
    a = Dense(3, activation='softmax')(a)#全连接层，3个隐藏神经元
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])#矩阵相乘，注意力得分计算
    return output_attention_mul

#自编码，解码器
def main_task(inputs, name):
    decode = Dense(dim)(inputs)
    decode = Reshape((4, 4, 16))(decode)
    decode = UpSampling2D((2, 2))(decode)#上采样，恢复为原来的形状
    decode = Conv2D(8, (3, 3), padding='same')(decode)
    decode = UpSampling2D((2, 2))(decode)
    decoder_output = Conv2D(1, (3, 3), padding='same', name=name + '_output')(decode)
    return decoder_output

#自编码，编码器
def encoder(inputs):
    encode = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(inputs)
    encode = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(encode)
    encode = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(encode)
    encode = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(encode)
    encode = Dropout(0.3)(encode)
    return encode

#辅助任务
def aux_task(inputs, name):
    aux_predict = Reshape((4, 4, 16))(inputs)
    aux_predict = Conv2D(16, (3, 3), padding='same', activation='relu')(aux_predict)
    aux_predict = UpSampling2D((2, 2))(aux_predict)
    aux_predict = Conv2D(8, (3, 3), padding='same', activation='relu')(aux_predict)
    aux_predict = UpSampling2D((2, 2))(aux_predict)
    aux_predict = Conv2D(1, (3, 3), padding='same', activation='relu', name=name+'_auxiliary')(aux_predict)
    return aux_predict

#数据组织（供给与需求）
[demandX_train, supplyX_train] = np.load('train.npz')['X']
[demandY_train, supplyY_train] = np.load('train.npz')['Y']
#影响因素数据输入
factor_train = np.load('train.npz')['factor']
#辅助任务训练数据
demand_aux_train = np.load('train.npz')['auxiliary'][:, :, :, :1]
supply_aux_train = np.load('train.npz')['auxiliary'][:, :, :, 1:]
#测试数据加载
[demandX_test, supplyX_test] = np.load('test.npz')['X']
[demandY_test, supplyY_test] = np.load('test.npz')['Y']
#影响因素测试数据
factor_test = np.load('test.npz')['factor']
#辅助任务测试数据
demand_aux_test = np.load('test.npz')['auxiliary'][:, :, :, :1]
supply_aux_test = np.load('test.npz')['auxiliary'][:, :, :, 1:]

timestep = 3
size = 16
dim = 4 * 4 * 16

input_demand = Input(shape=(None, size, size, 1))
demand_encoder = encoder(input_demand)
demand_reshape = TimeDistributed(Dropout(0.3))(demand_encoder)
demand_reshape = TimeDistributed(Flatten())(demand_reshape)
demand_reshape = Reshape((timestep, dim))(demand_reshape)
#supply的encoder，然后reshape到dim*timestep的形式

#GCN基本参数，以及邻接矩阵的处理
# Define parameters
FILTER = 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
#adj数据
adj = []
adj = nx.adjacency_matrix(G1)
#简单GCN模型邻接矩阵的预处理和元组表示的转换,A~=A+I,添加self-loop,度矩阵的归一化处理
#features数据格式构造
#input_supply
input_supply = Input(shape=(None, size, size, 1))
#将得到的张量转换为数组先进行feature的处理

# 将输入先经过GCN中feature的处理
features = BatchNormalization()(input_supply)
features = tf.reshape(features, [-1, 1])
#utils中的，行规范化特征矩阵并转换为元组表示
if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    support = 1
    graph = [features, adj]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    T_k = chebyshev_polynomial(adj, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = [features]+T_k
    G = [Input( batch_shape=(None, None), sparse=True)
         for _ in range(support)]
else:
    raise Exception('Invalid filter type.')
#将处理的feature数组再化为tensor神经网络的输出
input_supply = Lambda(lambda x: x)(features)
#卷积层结构堆叠
input_supply = Dropout(rate=0.5)(input_supply)
input_supply = GraphConvolution(256, support, activation='relu', kernel_regularizer=l2(5e-4))([input_supply]+G)
input_supply = Dropout(rate=0.5)(input_supply)
input_supply = GraphConvolution(256, support, activation='softmax')([input_supply]+G)
#在经过原先的处理输入至LSTM网络
input_supply = Lambda(lambda x: tf.reshape(x, [-1, -1, 16, 16, 1]))(input_supply)
#这里必须要转一下格式
supply_encoder = encoder(input_supply)
supply_reshape = TimeDistributed(Dropout(0.3))(supply_encoder)
supply_reshape = TimeDistributed(Flatten())(supply_reshape)
supply_reshape = Reshape((timestep, dim))(supply_reshape)
#需求CNN层处理
demand_decoder = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(demand_encoder)
demand_decoder = TimeDistributed(UpSampling2D((2, 2)))(demand_decoder)
demand_decoder = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(demand_decoder)
demand_decoder = TimeDistributed(UpSampling2D((2, 2)))(demand_decoder)
demand_decoder = TimeDistributed(Conv2D(1, (3, 3), padding='same', activation='relu'))(demand_decoder)

supply_decoder = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(supply_encoder)
supply_decoder = TimeDistributed(UpSampling2D((2, 2)))(supply_decoder)
supply_decoder = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(supply_decoder)
supply_decoder = TimeDistributed(UpSampling2D((2, 2)))(supply_decoder)
supply_decoder = TimeDistributed(Conv2D(1, (3, 3), padding='same', activation='relu'))(supply_decoder)


combine_demand_supply = concatenate([demand_reshape, supply_reshape])
lstm = LSTM(dim, return_sequences=1, input_shape=(timestep, dim * 2))(combine_demand_supply)

input_aux = Input(shape=(size, size, 13))
aux_encode = Conv2D(16, (3, 3), padding='same', activation='relu')(input_aux)
aux_encode = MaxPooling2D(pool_size=(2, 2))(aux_encode)
aux_encode = Conv2D(32, (3, 3), padding='same', activation='relu')(aux_encode)
aux_encode = MaxPooling2D(pool_size=(2, 2))(aux_encode)
aux_decode = Conv2D(32, (3, 3), padding='same', activation='relu')(aux_encode)
aux_decode = UpSampling2D((2, 2))(aux_decode)
aux_decode = Conv2D(16, (3, 3), padding='same', activation='relu')(aux_decode)
aux_decode = UpSampling2D((2, 2))(aux_decode)
aux_decode = Conv2D(13, (3, 3), padding='same', activation='relu', name='autoencoder')(aux_decode)


aux_dim = 32*4*4
aux = Reshape((aux_dim,))(aux_encode)

aux_demand = Dense(aux_dim)(aux)
aux_demand = Dense(dim)(aux_demand)
aux_demand = Dropout(0.5)(aux_demand)
aux_demand_predict = aux_task(aux_demand, 'demand')

aux_supply = Dense(aux_dim)(aux)
aux_supply = Dense(dim)(aux_supply)
aux_supply = Dropout(0.5)(aux_supply)
aux_supply_predict = aux_task(aux_supply, 'supply')


demand_attention = attention_3d_block(lstm)
demand_attention = Flatten()(demand_attention)
demand_attention = Dense(dim * 2)(demand_attention)
demand_combine = concatenate([demand_attention, aux_demand])
demand_combine = Dense(dim * 2)(demand_combine)
demand_predict = main_task(demand_combine, 'demand')

supply_attention = attention_3d_block(lstm)
supply_attention = Flatten()(supply_attention)
supply_attention = Dense(dim * 2)(supply_attention)
supply_combine = concatenate([supply_attention, aux_supply])
supply_combine = Dense(dim * 2)(supply_combine)
supply_predict = main_task(supply_combine, 'supply')

model = Model(inputs=[input_demand, input_supply, input_aux],
              outputs=[demand_predict, supply_predict, aux_decode, aux_demand_predict, aux_supply_predict, demand_decoder, supply_decoder])
model.compile(loss='mse',
              optimizer='adam',
              metrics=[rmse],
              loss_weights=[1, 1, 2, 2, 2, 0.25, 0.25])

print(model.summary())
# plot_model(model, to_file='model.png')

history = model.fit({demandX_train, supplyX_train, factor_train},#模型多输入
                    [demandY_train, supplyY_train, factor_train, demand_aux_train, supply_aux_train, demandX_train, supplyX_train],
                    batch_size=8,
                    epochs=100,
                    verbose=2,
                    validation_data=([demandX_test, supplyX_test, factor_test, graph],
                                     [demandY_test, supplyY_test, factor_test, demand_aux_test, supply_aux_test, demandX_test, supplyX_test]))

demand_rmse = history.history['demand_output_rmse']
val_demand_rmse = history.history['val_demand_output_rmse']
supply_rmse = history.history['supply_output_rmse']
val_supply_rmse = history.history['val_supply_output_rmse']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, demand_rmse, 'bo', label='Demand Training RMSE')
plt.plot(epochs, val_demand_rmse, 'b', label='Demand Validation RMSE')
plt.plot(epochs, supply_rmse, 'ro', label='Supply Training RMSE')
plt.plot(epochs, val_supply_rmse, 'r', label='Supply Validation RMSE')
plt.title('Training and validation RMSE')
plt.ylim(1, 3)
plt.xlim(1, epochs[-1])
plt.grid(1)
plt.axhline(2.3)
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
