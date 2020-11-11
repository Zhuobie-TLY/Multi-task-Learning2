from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


from graph88 import GraphConvolution
from utils88 import *
#from graph import *


#XX = supplyX_train[56][2]
#X = np.reshape(XX, 256, 1)
#y = np.reshape(XX, 256, 1)
# Define parameters
FILTER = 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 30  # early stopping patience

#X = np.zeros(256, 1)
# Get data
#X:features  A:graph  y:labels
X, A, y = load_data(dataset='cora', use_feature=True)
#X = np.ones((256, 1))
#adj数据
#adj = []
#adj = nx.adjacency_matrix(G1)
#数据组织（供给与需求）
#y_train = np.load("train.npz")["X"][1][:, 0].reshape([1536,256,1])
#y_val = np.load("train.npz")["Y"][1].reshape([1536,256,1])
#y_test = np.load("test.npz")["Y"][1].reshape([1536,256,1])
y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(y)

# Normalize X
#X /= X.sum(1).reshape(-1, 1)

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
    graph = [X]+T_k
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)
         for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

X_in = Input(shape=(X.shape[1],))


# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
H = Dropout(rate=0.5)(X_in)
H = GraphConvolution(16, support, activation='relu',
                     kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(rate=0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01), weighted_metrics=['acc'])

model.summary()



# Callbacks for EarlyStopping
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=PATIENCE)

# Train
#validation_data = (graph, y_val, val_mask)
validation_data = (graph, y_val)
model.fit(graph, y_train,
          batch_size=2708,
          epochs=NB_EPOCH,
          verbose=1,
          validation_data=validation_data,
          shuffle=False,
          callbacks=[es_callback])

# Evaluate model on the test data
#eval_results = model.evaluate(graph, y_test,
#                              sample_weight=test_mask,
#                              batch_size=A.shape[0])
eval_results = model.evaluate(graph, y_test,
                              batch_size=A.shape[0])
print('Test Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
