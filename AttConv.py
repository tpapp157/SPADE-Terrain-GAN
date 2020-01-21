
import tensorflow as tf


class AttentionConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.bias = bias
        self.stride = stride
        
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        
    def build(self, input_shape):
        self.rh = self.add_weight(shape=(1, 1, 1, self.kernel_size, 1, self.out_channels//2), initializer=tf.keras.initializers.RandomNormal(), dtype=tf.float32, name='rh')
        self.rw = self.add_weight(shape=(1, 1, 1, 1, self.kernel_size, self.out_channels//2), initializer=tf.keras.initializers.RandomNormal(), dtype=tf.float32, name='rh')
        
        self.q_conv = tf.keras.layers.Conv2D(self.out_channels, 1, strides=self.stride, padding='valid', kernel_initializer=tf.keras.initializers.GlorotUniform(), use_bias=self.bias, bias_initializer=tf.initializers.constant(0.0))
        self.k_conv = tf.keras.layers.Conv2D(self.out_channels, 1, strides=1, padding='valid', kernel_initializer=tf.keras.initializers.GlorotUniform(), use_bias=self.bias, bias_initializer=tf.initializers.constant(0.0))
        self.v_conv = tf.keras.layers.Conv2D(self.out_channels, 1, strides=1, padding='valid', kernel_initializer=tf.keras.initializers.GlorotUniform(), use_bias=self.bias, bias_initializer=tf.initializers.constant(0.0))
        
        
    def call(self, x):
        q = self.q_conv(x)
        
        batch = tf.shape(q)[0]
        height = tf.shape(q)[1]
        width = tf.shape(q)[2]
        #channels = tf.shape(x)[3]
        
        p = (self.kernel_size-1)//2
        x_pad = tf.pad(x, [[0,0], [p,p], [p,p], [0,0]], mode='SYMMETRIC')
        
        k = self.k_conv(x_pad)
        v = self.v_conv(x_pad)
        
        k = tf.image.extract_patches(k, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1,self.stride,self.stride,1], rates=[1,1,1,1], padding='VALID')
        v = tf.image.extract_patches(v, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1,self.stride,self.stride,1], rates=[1,1,1,1], padding='VALID')
        
        v1, v2 = tf.split(tf.reshape(v, [batch, height, width, self.kernel_size, self.kernel_size, self.out_channels]), 2, axis=-1)
        v = tf.concat([v1+self.rh, v2+self.rw], axis=-1)
        
        k = tf.reshape(k, (batch, self.groups, self.out_channels//self.groups, height, width, -1))
        v = tf.reshape(v, (batch, self.groups, self.out_channels//self.groups, height, width, -1))
        q = tf.reshape(q, (batch, self.groups, self.out_channels//self.groups, height, width, 1))
        
        out = q * k
        out = tf.nn.softmax(out, -1)
        out = tf.reshape(tf.einsum('bnchwk,bnchwk -> bnchw', out, v), (batch, height, width, self.out_channels))
        
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape[:3] + self.out_channels
