from __future__ import print_function

from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from graph8 import GraphConvolution
from utils8 import *
from graph import *
from matplotlib import pyplot as plt


# Define parameters
FILTER = 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 50
PATIENCE = 30  # early stopping patience


# Get data
#X:features  A:graph  y:labels
#X, A, y = load_data(dataset='cora', use_feature=True)
#X = np.ones((256, 1))
#adj数据
A = []
A = nx.adjacency_matrix(G1)
#数据组织（供给与需求）
X = np.load("train.npz")["X"][1][:, 0].reshape([1536,256,1])
y_train = np.load("train.npz")["Y"][1].reshape([1536,256,1])
y_val = np.load("train.npz")["Y"][1].reshape([1536,256,1])
#y_test = np.load("test.npz")["Y"][1].reshape([1536,256,1])
#y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(y)

# Normalize X
#X[2] /=X.sum(2).reshape(-1, 1)
#y_train[2] /= y_train.sum(2).reshape(-1, 1)
#y_val[2] /= y_val.sum(2).reshape(-1, 1)

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    # 得到对称规范化的图拉普拉斯矩阵，L = I - D ^ (-1/2) * A * D ^ (-1/2)
    #A coomatrix,2707*2707
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    T_kk = list()
    for j in range(support):
        #将sparsematrix转换为matrix
        T1 = T_k[j].todense()
        T2 = T1.A
        #matrix添加纬度
        T3 = np.zeros(shape=(1536, 256, 256))
        T4 = T2+T3
        #再压回去
        T_kk.append(T4)


    #T_k = np.reshape(T_k, (3, 256, 256, 1))
    print(T_kk[0].shape)
    graph = [X]+T_kk
    G = [Input(shape=(None, 3), batch_shape=(None, 256, 256), sparse=False)for _ in range(support)]
    # 一个尺寸元组（整数），包含批量大小。 例如，batch_shape=(10, 32)
    # 表明期望的输入是 10 个 32 维向量。 batch_shape=(None, 32) 表明任意批次大小的 32 维向量。

else:
    raise Exception('Invalid filter type.')

X_in = Input(shape=(X.shape[1], 1))


# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
H = Dropout(rate=0.5)(X_in)
#修改相乘的维度，此处变为[N, N]*[?, N, N]
H = GraphConvolution(16, support, activation='relu',
                     kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(rate=0.5)(H)
Y = GraphConvolution(y_train.shape[2], support, activation='relu')([H]+G)#(?,256,1)
# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)

model.compile(loss='mse',
              optimizer=Adam(lr=0.01), weighted_metrics=['acc'])

model.summary()



# Callbacks for EarlyStopping
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=PATIENCE)

# Train

validation_data = (graph, y_val)
history = model.fit(graph, y_train,
          batch_size=30,
          epochs=NB_EPOCH,
          verbose=1,
          validation_data=validation_data,
          shuffle=False,
          callbacks=[es_callback])

#可视化部分

history = history
plt.plot()
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import numpy as np

accy = history.history['acc']
np_accy = np.array(accy)
np.savetxt('save_acc.txt', np_accy)

# Evaluate model on the test data
#eval_results = model.evaluate(graph, y_test,
#                              sample_weight=test_mask,
#                              batch_size=A.shape[0])
#eval_results = model.evaluate(graph, y_test,
#                              batch_size=27)
#print('Test Done.\n'
#      'Test loss: {}\n'
#      'Test accuracy: {}'.format(*eval_results))
