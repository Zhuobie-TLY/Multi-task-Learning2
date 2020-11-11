import numpy as np
import tensorflow as tf
from keras.layers import Permute, Dense, TimeDistributed, Conv2D, MaxPooling2D, Multiply, Dropout, Flatten, Reshape, \
    LSTM, UpSampling2D, concatenate, Input, GraphConvolution
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from graph import *
from utils8 import *
from matplotlib import pyplot as plt
# from keras.utils import plot_model

#输入是supplyX_train1[1536, 256, 1]，要求最后输出的是graph1[(1536, 256, 1), (1536, 256, 256), (1536, 256, 256)]
def Graphplus(x, T_kk1, T_kk2):
    GCN1 = []
    GCN2 = []
    GCN3 = []
    for i in range(x.shape[0]):
            a = divmod(i, 96)[1]
            if 28 < a <= 36 or 68 < a <= 76:
                GCN1.append(T_kk1[0])
                GCN2.append(T_kk1[1])
                GCN3.append(T_kk1[2])
            else:
                GCN1.append(T_kk2[0])
                GCN2.append(T_kk2[1])
                GCN3.append(T_kk2[2])
    GCN1 = np.array(GCN1)
    GCN2 = np.array(GCN2)
    GCN3 = np.array(GCN3)
    print(x.shape)
    print(GCN1.shape)
    graph = [x, GCN1, GCN2, GCN3]
    return graph

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

#这个函数实现输入adj,然后进行切比雪夫fliter的处理，输出T_kk[256, 256]
def graphprocess(G, MAX_DEGREE):
    # GCN超参数以及数据准备,计算T_kk2, T_kk2
    # Define parameters
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    support = MAX_DEGREE+1
    # graph数据
    # adj数据
    A = []
    A = nx.adjacency_matrix(G)
    # fliter
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    # 得到对称规范化的图拉普拉斯矩阵，L = I - D ^ (-1/2) * A * D ^ (-1/2)
    # A coomatrix,2707*2707
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    T_kk = []
    for j in range(support):
        T1 = T_k[j].todense()
        T2 = T1.A
        T_kk.append(T2)
    return T_kk

def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(3, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def main_task(inputs, name):
    decode = Dense(dim)(inputs)
    decode = Reshape((4, 4, 16))(decode)
    decode = UpSampling2D((2, 2))(decode)
    decode = Conv2D(8, (3, 3), padding='same')(decode)
    decode = UpSampling2D((2, 2))(decode)
    decoder_output = Conv2D(1, (3, 3), padding='same', name=name + '_output')(decode)
    return decoder_output


def encoder(inputs):
    encode = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(inputs)
    encode = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(encode)
    encode = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(encode)
    encode = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(encode)
    encode = Dropout(0.3)(encode)
    return encode


def aux_task(inputs, name):
    aux_predict = Reshape((4, 4, 16))(inputs)
    aux_predict = Conv2D(16, (3, 3), padding='same', activation='relu')(aux_predict)
    aux_predict = UpSampling2D((2, 2))(aux_predict)
    aux_predict = Conv2D(8, (3, 3), padding='same', activation='relu')(aux_predict)
    aux_predict = UpSampling2D((2, 2))(aux_predict)
    aux_predict = Conv2D(1, (3, 3), padding='same', activation='relu', name=name+'_auxiliary')(aux_predict)
    return aux_predict


[demandX_train, supplyX_train] = np.load('train.npz')['X']
[demandY_train, supplyY_train] = np.load('train.npz')['Y']
#供给数据再组织
supplyX_train = np.load("train.npz")["X"][1].reshape([1536, 3, 256, 1])
factor_train = np.load('train.npz')['factor']
demand_aux_train = np.load('train.npz')['auxiliary'][:, :, :, :1]
supply_aux_train = np.load('train.npz')['auxiliary'][:, :, :, 1:]
[demandX_test, supplyX_test] = np.load('test.npz')['X']
[demandY_test, supplyY_test] = np.load('test.npz')['Y']
supplyX_test = np.load("test.npz")["X"][1].reshape([672, 3, 256, 1])
factor_test = np.load('test.npz')['factor']
demand_aux_test = np.load('test.npz')['auxiliary'][:, :, :, :1]
supply_aux_test = np.load('test.npz')['auxiliary'][:, :, :, 1:]

timestep = 3
size = 16
dim = 4 * 4 * 16
#将supply_train分为三块
supplyX_train = supplyX_train.transpose((1, 0, 2, 3))
supplyX_train1 = supplyX_train[0]
supplyX_train2 = supplyX_train[1]
supplyX_train3 = supplyX_train[2]
#将supply_test分为三块
supplyX_test = supplyX_test.transpose((1, 0, 2, 3))
supplyX_test1 = supplyX_test[0]
supplyX_test2 = supplyX_test[1]
supplyX_test3 = supplyX_test[2]


input_demand = Input(shape=(None, size, size, 1))
demand_encoder = encoder(input_demand)
demand_reshape = TimeDistributed(Dropout(0.3))(demand_encoder)
demand_reshape = TimeDistributed(Flatten())(demand_reshape)
demand_reshape = Reshape((timestep, dim))(demand_reshape)
"""
demand_decoder = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(demand_encoder)
demand_decoder = TimeDistributed(UpSampling2D((2, 2)))(demand_decoder)
demand_decoder = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(demand_decoder)
demand_decoder = TimeDistributed(UpSampling2D((2, 2)))(demand_decoder)
demand_decoder = TimeDistributed(Conv2D(1, (3, 3), padding='same', activation='relu'))(demand_decoder)
"""
#相关参数
MAX_DEGREE = 2  # maximum polynomial degree
support = MAX_DEGREE + 1
T_k1 = graphprocess(G=G1, MAX_DEGREE=MAX_DEGREE)#GF graph
T_k2 = graphprocess(G=G2, MAX_DEGREE=MAX_DEGREE)#PF graph
##带graph的sample处理
graph1 = Graphplus(supplyX_train1, T_k1, T_k2)
graph2 = Graphplus(supplyX_train2, T_k1, T_k2)
graph3 = Graphplus(supplyX_train3, T_k1, T_k2)
graph1t = Graphplus(supplyX_test1, T_k1, T_k2)
graph2t = Graphplus(supplyX_test2, T_k1, T_k2)
graph3t = Graphplus(supplyX_test2, T_k1, T_k2)

#G数据输入
G11 = [Input(batch_shape=(None, 256, 256), sparse=False)for _ in range(support)]
G22 = [Input(batch_shape=(None, 256, 256), sparse=False) for _ in range(support)]
G33 = [Input(batch_shape=(None, 256, 256), sparse=False) for _ in range(support)]
# 一个尺寸元组（整数），包含批量大小。 例如，batch_shape=(10, 32)
# 表明期望的输入是 10 个 32 维向量。 batch_shape=(None, 32) 表明任意批次大小的 32 维向量。

#supply_GCN层处理
#1
X_in1 = Input(shape=(supplyX_train1.shape[1], 1))#[None, 256, 1]# Define model architecture# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
H1 = Dropout(rate=0.5)(X_in1)#修改相乘的维度，此处变为[N, N]*[?, N, N]#自定义层，将[(1536, 3, 256, 1), (1536, 3, 256, 256), (1536, 3, 256, 256), (1536, 3, 256, 256) ]#每个list调整维度，变为#concentrate函数转变为(A1, A2, A3)tuple类型其中A为((1536, 256, 1), (1536, 256, 256),(1536, 256, 256))
H1 = GraphConvolution(16, support, activation='relu',
                     kernel_regularizer=l2(5e-4))([H1]+G11)
H1 = Dropout(rate=0.5)(H1)
Y1 = GraphConvolution(1, support, activation='relu')([H1]+G11)#[None, 256, 1]new[None, None, 256, 1]

#2
X_in2 = Input(shape=(supplyX_train2.shape[1], 1))#[None, 256, 1]# Define model architecture# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
H2 = Dropout(rate=0.5)(X_in2)#修改相乘的维度，此处变为[N, N]*[?, N, N]#自定义层，将[(1536, 3, 256, 1), (1536, 3, 256, 256), (1536, 3, 256, 256), (1536, 3, 256, 256) ]#每个list调整维度，变为#concentrate函数转变为(A1, A2, A3)tuple类型其中A为((1536, 256, 1), (1536, 256, 256),(1536, 256, 256))
H2 = GraphConvolution(16, support, activation='relu',
                     kernel_regularizer=l2(5e-4))([H2]+G22)
H2 = Dropout(rate=0.5)(H2)
Y2 = GraphConvolution(1, support, activation='relu')([H2]+G22)#[None, 256, 1]new[None, None, 256, 1]


#3
X_in3 = Input(shape=(supplyX_train3.shape[1], 1))#[None, 256, 1]# Define model architecture# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
H3 = Dropout(rate=0.5)(X_in3)#修改相乘的维度，此处变为[N, N]*[?, N, N]#自定义层，将[(1536, 3, 256, 1), (1536, 3, 256, 256), (1536, 3, 256, 256), (1536, 3, 256, 256) ]#每个list调整维度，变为#concentrate函数转变为(A1, A2, A3)tuple类型其中A为((1536, 256, 1), (1536, 256, 256),(1536, 256, 256))
H3 = GraphConvolution(16, support, activation='relu',
                     kernel_regularizer=l2(5e-4))([H3]+G33)
H3 = Dropout(rate=0.5)(H3)
Y3 = GraphConvolution(1, support, activation='relu')([H3]+G33)#[None, 256, 1]new[None, None, 256, 1]

#三个合起来
Y = concatenate([Y1, Y2, Y3], axis=-1)
supply_reshape = Reshape((timestep, dim))(Y)#[None, 3, 256]
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

model = Model(inputs=[input_demand, input_aux]+[X_in1]+G11+[X_in2]+G22+[X_in3]+G33,
              outputs=[demand_predict, supply_predict, aux_decode, aux_demand_predict, aux_supply_predict])
model.compile(loss='mse',
              optimizer='adam',
              metrics=[rmse],
              loss_weights=[1, 1, 10, 20, 20])

print(model.summary())
# plot_model(model, to_file='model.png')

history = model.fit([demandX_train, factor_train]+graph1+graph2+graph3,
                    [demandY_train, supplyY_train, factor_train, demand_aux_train, supply_aux_train],
                    batch_size=8,
                    epochs=120,
                    verbose=2,
                    validation_data=([demandX_test, factor_test]+graph1t+graph2t+graph3t,
                                     [demandY_test, supplyY_test, factor_test, demand_aux_test, supply_aux_test]))

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
print(val_demand_rmse)
print(val_supply_rmse)
