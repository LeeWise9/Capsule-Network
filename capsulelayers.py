import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


class Length(layers.Layer):
    """
    计算向量的长度. 这个长度可以用于预测类别, 最长的向量对应的index就是预测的类别, 
    计算原理类似于 `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
    对具有这种形状 [None, num_capsule, dim_vector] 的 Tensor 进行蒙版操作. 提供两种方式:
    1. 将向量长度为最大的向量值保留, 其余的值设置为0, 最后flatten为一维向量
    2. 使用输入的蒙版进行如上操作
    举例:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, 每一条数据包含 3 个胶囊, 每一个胶囊向量长度为 2
        out = Mask()(x)  # out.shape=[8, 6] 输出的尺寸 [8, 6]
        # or
        y = keras.layers.Input(shape=[8, 3])  # 真实标签. 8 条数据, 3 个类别对应 3 个胶囊, 每一条数据为独热变量的向量(one-hot coding).
        out2 = Mask()([x, y])  # out2.shape=[8,6]. 使用给定的 y 作为蒙版进行操作
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # 如果输入包含蒙版, 形状为 [None, n_classes], 每一条数据使用独热编码(one-hot code).
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # 如果输入没有包含蒙版, 使用胶囊向量长度产生蒙版. 
            # 计算胶囊长度
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # 产生蒙版, 为独热编码
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # 如果输入包含蒙版的情况
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # 如果输入不包含蒙版的情况
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    对向量的非线性激活函数
    ## vectors: some vectors to be squashed, N-dim tensor
    ## axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    胶囊层. 输入输出都为向量. 
    ## num_capsule: 本层包含的胶囊数量
    ## dim_capsule: 输出的每一个胶囊向量的维度
    ## routings: routing 算法的迭代次数
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_initializer='glorot_uniform',**kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, '输入的 Tensor 的形状[None, input_num_capsule, input_dim_capsule]'
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        #转换矩阵
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                initializer=self.kernel_initializer,name='W')
        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsuie, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # 运算优化:将inputs_expand重复num_capsule 次，用于快速和W相乘
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # 将inputs_tiled的batch中的每一条数据，计算inputs+W
        # x.shape = [num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape = [num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # 将x和W的前两个维度看作'batch'维度，向量和矩阵相乘:
        # [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsutel
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]),elems=inputs_tiled)

        # Begin: Routing算法
        # 将系数b初始化为0.
        # b.shape = [None, self.num_capsule, self, input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])
        
        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[None, num_capsule, input_num_capsule]
            C = tf.nn.softmax(b, dim=1)
            # c.shape = [None, num_capsule, input_num_capsule]
            # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
            # 将c与inputs_hat的前两个维度看作'batch'维度，向量和矩阵相乘:
            # [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule],
            # outputs.shape= [None, num_capsule, dim_capsule]
            outputs = squash(K. batch_dot(C, inputs_hat, [2, 2])) # [None, 10, 16]
        
            if i < self.routings - 1:
                # outputs.shape = [None, num_capsule, dim_capsule]
                # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
                # 将outputs和inρuts_hat的前两个维度看作‘batch’ 维度，向量和矩阵相乘:
                # [dim_capsule] x [imput_num_capsule, dim_capsule]^T -> [input_num_capsule]
                # b.shape = [batch_size. num_capsule, input_nom_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing 算法
        return outputs


    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
            }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    进行普通二维卷积 `n_channels` 次, 然后将所有的胶囊重叠起来
    ## inputs: 4D tensor, shape=[None, width, height, channels]
    ## dim_capsule: the dim of the output vector of capsule
    ## n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, 
                           strides=strides, padding=padding, name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

