
import tensorflow as tf
import numpy as np

# =============================================================================
# def conv(x, channels, kernel=3, stride=1, pad=0, pad_type='symmetric', use_bias=True):
#     if kernel>1:
#         p = (kernel-1)//2
#         x = tf.pad(x, [[0,0], [p,p], [p,p], [0,0]], mode='SYMMETRIC')
# 
#     x = tf.keras.layers.Conv2D(channels, kernel, strides=stride, padding='valid', kernel_initializer=tf.keras.initializers.GlorotUniform(), use_bias=use_bias, bias_initializer=tf.initializers.constant(0.0))(x)
# 
#     return x
# =============================================================================


class ScaledSum(tf.keras.layers.Layer):
    def __init__(self, size):
        super(ScaledSum, self).__init__()
        self.size = size
        
    def build(self, input_shape):
        #size = tf.math.minimum(input_shape[1], input_shape[2])
        assert self.size>=8, "ScaledSum image dimensions must be >=8"
        
        self.mask = []
        for i in np.arange(2, np.log2(self.size), 1, dtype='int'):
            k = int(self.size**2//4**(i+1))
            temp = tf.transpose(tf.repeat(tf.repeat(tf.reshape(tf.eye(k), (self.size//2**(i+1), self.size//2**(i+1), k)), 2**(i+1), axis=0), 2**(i+1), axis=1), (2,0,1))
            temp = tf.expand_dims(tf.expand_dims(temp, 0), -1) * (tf.size(temp, out_type=tf.float32) / tf.math.reduce_sum(temp))
            self.mask.append(temp)
        
        self.a = self.add_weight(shape=(1, 1, 1, len(self.mask)+1, 1), initializer=tf.keras.initializers.ones(), dtype=tf.float32, name='a')
        
        
    def call(self, x):
        batch = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]
        
        out = [tf.expand_dims(x, -2)]
        x = tf.expand_dims(x, 1)
        for m in self.mask:
            i = tf.cast(tf.sqrt(tf.cast(tf.shape(m)[1], dtype=tf.float32)), tf.int32)
            temp = tf.reshape(tf.math.reduce_mean(x * m, axis=(2,3)), (batch, i, i, channels))
            temp = tf.expand_dims(tf.repeat(tf.repeat(temp, height//i, axis=1), width//i, axis=2), -2)
            out.append(temp)
            #out.append(tf.expand_dims(tf.repeat(tf.reshape(x * m, (batch, i, i, channels)), (1, height//i, width//i, 1)), -2))
        out = tf.reduce_mean(tf.concat(out, axis=-2) * tf.nn.softmax(self.a), axis=-2)
        
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape