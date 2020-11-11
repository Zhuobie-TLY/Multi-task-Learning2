from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K


class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, #[N,F]+[N,F]
                 units,  #16
                 support=1,#3
                 activation=None,#'RELU'
                 use_bias=True,
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
        # 特征矩阵形状#[N,F]
        features_shape = input_shapes[0]#[None,F]
        # 输出形状为(批大小, 输出维度)[B, 16]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)[B, 16]
     #input shape[(None, 1433),(None,None),(None,None),(None, None)]
    def build(self, input_shapes):#这是定义权重的方法，可训练的权应该在这里被加入列
        features_shape = input_shapes[0]#[None,1433]
        assert len(features_shape) == 2
        input_dim = features_shape[1]#1433

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),#(F*3, 16)
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),#(16,)
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):#这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量
        features = inputs[0]#[N,F]
        basis = inputs[1:]#[(None,None),(None,None),(None,None)]

        supports = list()#(?,4399)
        for i in range(self.support):
            #A*X
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        #A*X*W
        output = K.dot(supports, self.kernel)

        if self.bias:
            #A*X*W+b
            output += self.bias
        return self.activation(output)#非线性激活

