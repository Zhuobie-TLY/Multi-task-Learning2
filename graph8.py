from __future__ import print_function
import tensorflow as tf
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import scipy.sparse as sp

class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, #[N,F]+[N,F]新[（？， 256， 1），（?,256,256），（?,256,256），（?,256,256）]
                 units,  #16
                 support=1,#3
                 activation=None,#'RELU'
                 use_bias=None,
                 kernel_initializer='glorot_uniform', #Gaussian distribution L2(5e-4)
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # 施加在权重上的正则项
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # 施加在偏置向量上的正则项
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # 施加在输出上的正则项
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # 对主权重矩阵进行约束
        self.kernel_constraint = constraints.get(kernel_constraint)
         # 对偏置向量进行约束
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):#如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断
        # 特征矩阵形状#[N,F]新[?,255,1]
        features_shape = input_shapes[0]#[None,F]新[?, N, F][?,255,1]
        # 输出形状为(批大小, 输出维度)[B, 16]新[Batch, 256, 16]
        output_shape = (features_shape[0], 256, self.units)
        return output_shape  # (batch_size, output_dim)[B, 16]新[Batch, 256, 16]
     #input shape[(None, 1433),(None,None),(None,None),(None, None)]
     #新input shape[(None, 256，1),(None,None),(None,None),(None, None)]
    def build(self, input_shapes):#这是定义权重的方法，可训练的权应该在这里被加入列
        features_shape = input_shapes[0]#[None,1433]新[None, 256，1]
        assert len(features_shape) == 3
        input_dim = features_shape[2]#1433新1

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),#(1433*3, 16)新(1*3, 16)
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(256, self.units,),#(？，16)新（？，256，16）
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):#这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量
        #现将[(?,256,256),(?,256,256),(?,256,256)]转变为[(256, 256), (256, 256), (256, 256)]
        basis = list()
        for k in range(1, 4):
            B1 = inputs[k][0]
            basis.append(B1)
        features = inputs[0]#[N,F][?, F]新[?,256,1]

        supports = list()#(?,4399)
        for i in range(self.support):
            #A*X
            #old: [2708,2708]*[2708,1433]=[2708, 1433][?,1433]
            #NEW: [256,256]*[?,256,1]=[?,256,1]
            AAA = basis[i]
            AX = tf.einsum("ij,ljk->lik", AAA, features)
            supports.append(AX)
        supports = K.concatenate(supports, axis=2)
        #old:[?,4299]
        #new:[?, 256, 3]
        #A*X*W
        output = tf.einsum("lij,jk->lik", supports, self.kernel)
        #old:(?, 16)，supports=[?,4399][2708,4399],self.kernel[4399, 16]
        #new:(？，256，16),supports=[?,256,3],self.kernel[3,16],要得到的是[?,256,16]
        if self.bias:
            #A*X*W+b
            output += self.bias
            #old:(?, 16),(?,16)+(?,16)
            #new:(?, 256,16),(?, 256,16)+(?, 256,16)
        return self.activation(output)#非线性激活

