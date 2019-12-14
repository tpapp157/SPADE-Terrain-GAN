from __future__ import absolute_import, division, print_function, unicode_literals

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

#from VGGLoss import VGGLoss

#%%
PATH = r'datasets\TERR\trainH'

#%%
BUFFER_SIZE = 50
BATCH_SIZE = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256



def load_train(inf, trf, htf):
    C = {(17, 141, 215): 0.0,
         (225, 227, 155): 1.0,
         (127, 173, 123): 2.0,
         (185, 122, 87): 3.0,
         (230, 200, 181): 4.0,
         (150, 150, 150): 5.0,
         (193, 190, 175): 6.0}
    
    C = np.array([[17, 141, 215],[225, 227, 155],[127, 173, 123],[185, 122, 87],[230, 200, 181],[150, 150, 150],[193, 190, 175]])
    C = np.reshape(C, (1,1,C.shape[0],3))
    
    input_image = tf.io.decode_png(tf.io.read_file(inf))
    real_image = tf.io.decode_png(tf.io.read_file(trf))
    height_image = tf.io.decode_png(tf.io.read_file(htf), dtype=tf.uint16)
    
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    height_image = tf.cast(height_image, tf.float32)
    
    temp = tf.image.random_crop(tf.stack([input_image, real_image, tf.concat([height_image,tf.zeros_like(height_image),tf.zeros_like(height_image)], axis=2)], axis=0), size=[3, 256, 256, 3])
    input_image = temp[0]
    real_image = temp[1]
    height_image = tf.expand_dims(temp[2,:,:,0], -1)
    
    input_image = tf.image.resize(input_image, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [128, 128], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    height_image = tf.image.resize(height_image, [128, 128], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    
    input_image = tf.one_hot(tf.argmin(tf.norm(tf.expand_dims(input_image, -2)-C, axis=3), 2), C.shape[2], dtype=tf.float32)
    real_image = real_image / 127.5 - 1
    height_image = height_image / 32767.5 - 1
    
    return input_image, real_image, height_image


def load_test(inf, trf, htf):
    C = {(17, 141, 215): 0.0,
         (225, 227, 155): 1.0,
         (127, 173, 123): 2.0,
         (185, 122, 87): 3.0,
         (230, 200, 181): 4.0,
         (150, 150, 150): 5.0,
         (193, 190, 175): 6.0}
    
    C = np.array([[17, 141, 215],[225, 227, 155],[127, 173, 123],[185, 122, 87],[230, 200, 181],[150, 150, 150],[193, 190, 175]])
    C = np.reshape(C, (1,1,C.shape[0],3))
    
    input_image = tf.io.decode_png(tf.io.read_file(inf))
    real_image = tf.io.decode_png(tf.io.read_file(trf))
    height_image = tf.io.decode_png(tf.io.read_file(htf), dtype=tf.uint16)
    
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    height_image = tf.cast(height_image, tf.float32)
    
    temp = tf.image.random_crop(tf.stack([input_image, real_image, tf.concat([height_image,tf.zeros_like(height_image),tf.zeros_like(height_image)], axis=2)], axis=0), size=[3, 256, 256, 3])
    input_image = temp[0]
    real_image = temp[1]
    height_image = tf.expand_dims(temp[2,:,:,0], -1)
    
    input_image = tf.image.resize(input_image, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [128, 128], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    height_image = tf.image.resize(height_image, [128, 128], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    
    oh_image = tf.one_hot(tf.argmin(tf.norm(tf.expand_dims(input_image, -2)-C, axis=3), 2), C.shape[2], dtype=tf.float32)
    
    return oh_image, real_image, height_image, input_image


def load_file(inf):
    C = {(17, 141, 215): 0.0,
         (225, 227, 155): 1.0,
         (127, 173, 123): 2.0,
         (185, 122, 87): 3.0,
         (230, 200, 181): 4.0,
         (150, 150, 150): 5.0,
         (193, 190, 175): 6.0}
    
    C = np.array([[17, 141, 215],[225, 227, 155],[127, 173, 123],[185, 122, 87],[230, 200, 181],[150, 150, 150],[193, 190, 175]])
    C = np.reshape(C, (1,1,C.shape[0],3))
    
    input_image = tf.io.decode_png(tf.io.read_file(inf))[:,:,:3]
    input_image = tf.cast(input_image, tf.float32)
    oh_image = tf.one_hot(tf.argmin(tf.norm(tf.expand_dims(input_image, -2)-C, axis=3), 2), C.shape[2], dtype=tf.float32)
    
    return oh_image


#%%
files = glob.glob(os.path.join(PATH, '*_i2.png'))
tfiles = [f.replace('_i2.png', '_t.png') for f in files]
hfiles = [f.replace('_i2.png', '_h.png') for f in files]
#train_dataset = tf.data.Dataset.list_files(PATH+'*_i2.png')
train_dataset = tf.data.Dataset.from_tensor_slices((files, tfiles, hfiles))
train_dataset = train_dataset.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

#test_dataset = tf.data.Dataset.list_files(PATH+'test/*.png')
test_dataset = tf.data.Dataset.from_tensor_slices((files, tfiles, hfiles))
test_dataset = test_dataset.map(load_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(5)


#%%
def conv(x, channels, kernel=3, stride=1, pad=0, pad_type='symmetric', use_bias=True):
    if kernel>1:
        p = (kernel-1)//2
        x = tf.pad(x, [[0,0], [p,p], [p,p], [0,0]], mode='SYMMETRIC')

    x = tf.keras.layers.Conv2D(channels, kernel, strides=stride, padding='valid', kernel_initializer=tf.keras.initializers.GlorotUniform(), use_bias=use_bias, bias_initializer=tf.initializers.constant(0.0))(x)

    return x

#%%
class Add_Noise(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.b = self.add_weight(shape=(1, 1, 1, input_shape[3]), initializer='zeros', dtype=tf.float32, name='b')
    
    def call(self, x):
        return x + self.b * tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0, name='spade_rn')
    

def spade_resblock(segmap, x_init, channels, use_bias=True):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    x = spade(segmap, x_init, channel_in, use_bias=use_bias)
    
    x = Add_Noise()(x)
    
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias)

    x = spade(segmap, x, channels=channel_middle, use_bias=use_bias)
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias)

    if channel_in != channels :
        x_init = spade(segmap, x_init, channels=channel_in, use_bias=use_bias)
        x_init = conv(x_init, channels=channels, kernel=1, stride=1, use_bias=False)

    return x + x_init


def spade(segmap, x_init, channels, use_bias=True):
    x = param_free_norm(x_init)

    _, x_h, x_w, _ = tf.shape(x_init)
    _, segmap_h, segmap_w, _ = tf.shape(segmap)

    segmap_down = tf.image.resize(segmap, [x_h, x_w], method=tf.image.ResizeMethod.BILINEAR)

    segmap_down = conv(segmap_down, channels=128, kernel=5, stride=1, use_bias=use_bias)
    segmap_down = tf.nn.relu(segmap_down)

    segmap_gamma = conv(segmap_down, channels=channels, kernel=5, stride=1, use_bias=use_bias)
    segmap_beta = conv(segmap_down, channels=channels, kernel=5, stride=1, use_bias=use_bias)

    x = x * (1 + segmap_gamma) + segmap_beta
    return x


def param_free_norm(x, epsilon=1e-5):
    x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    x_std = tf.sqrt(x_var + epsilon)
    return (x - x_mean) / x_std


#%%
def simple_spade(x, segmap, out_ch):
    xc = x.get_shape().as_list()[-1]
    x = spade(segmap, x, xc)
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv(x, channels=out_ch, kernel=3, stride=1)
    return x

def con_conv(x, segmap, ch):
    _, x_h, x_w, _ = tf.shape(x)
    segmap = tf.image.resize(segmap, [x_h, x_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #segmap = conv(segmap, 64, 3)
    
    x = tf.nn.leaky_relu(x)
    x = tf.concat([x, segmap], axis=3)
    x = conv(x, channels=ch, kernel=3, stride=1)
    return x
    

#%%
OUTPUT_CHANNELS = 4


def Generator():
    channel = 768
    segmap = tf.keras.layers.Input(shape=[None,None,7])
    #batch_size = segmap.get_shape().as_list()[0]
    batch_size = tf.shape(segmap)[0]
    x = tf.random.uniform(shape=[batch_size, 16], dtype='float32')
    
    z_width = tf.shape(segmap)[2] // 2**5
    z_height = tf.shape(segmap)[1] // 2**5
    

    """
    # If num_up_layers = 5 (normal)
    
    # 64x64 -> 2
    # 128x128 -> 4
    # 256x256 -> 8
    # 512x512 -> 16
    
    """

    x = tf.keras.layers.Dense(channel)(x)
    x = tf.reshape(x, [batch_size, 1, 1, channel])
    x = tf.tile(x, [1, z_height, z_width, 1])
    #x = tf.image.resize(x, [z_height, z_width], method=tf.image.ResizeMethod.BILINEAR)
    

    x = spade_resblock(segmap, x, channels=channel, use_bias=True)
    #x = simple_spade(x, segmap, channel)
    #x = con_conv(x, segmap, channel)

    sf = 2
    x = tf.image.resize(x, [sf*z_height, sf*z_width], method=tf.image.ResizeMethod.BILINEAR)
    x = spade_resblock(segmap, x, channels=channel, use_bias=True)
    x = spade_resblock(segmap, x, channels=channel, use_bias=True)
    #x = simple_spade(x, segmap, channel)
    #x = simple_spade(x, segmap, channel)
    #x = con_conv(x, segmap, channel)
    #x = con_conv(x, segmap, channel)

    for i in range(4):
        sf = 2*sf
        channel = channel // 2
        x = tf.image.resize(x, [sf*z_height, sf*z_width], method=tf.image.ResizeMethod.BILINEAR)
        x = spade_resblock(segmap, x, channels=channel, use_bias=True)
        #x = simple_spade(x, segmap, channel)
        #x = con_conv(x, segmap, channel)
        
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv(x, channels=OUTPUT_CHANNELS, kernel=1, stride=1, use_bias=True)

    return tf.keras.Model(inputs=segmap, outputs=x)
generator = Generator()



def Shadow_Gen():
    inputs = tf.keras.layers.Input(shape=[None,None,4])
    T = inputs
    H = tf.tanh(tf.expand_dims(inputs[:,:,:,3], -1))
    
    top = H[:,1:,:,:]-H[:,:-1,:,:]
    top = tf.pad(top, [[0,0], [1,0], [0,0], [0,0]], mode='SYMMETRIC', name='top_pad')
    left = H[:,:,1:,:]-H[:,:,:-1,:]
    left = tf.pad(left, [[0,0], [0,0], [1,0], [0,0]], mode='SYMMETRIC', name='left_pad')
    H = tf.concat([top, left], axis=3)
    
    #H = tf.pad(H, [[0,0], [1,1], [1,1], [0,0]], mode='SYMMETRIC')
    #H = tf.keras.layers.Conv2D(8, 3, strides=1, kernel_initializer=initializer, use_bias=True, padding='VALID', activation=tf.nn.elu)(H)
    #H = tf.keras.layers.Conv2D(1, 1, strides=1, kernel_initializer=initializer, use_bias=True, padding='VALID')(H)
    H = tf.nn.leaky_relu(conv(H, 32, 3, 1))
    H = conv(H, 1, 1, 1)
    H = tf.concat([H, tf.zeros_like(H), tf.zeros_like(H), tf.zeros_like(H)], axis=3)
    out = tf.tanh(T+H)
    
    return tf.keras.Model(inputs=inputs, outputs=out)

shadow_generator = Shadow_Gen()

#%%
def lab_preprocess(lab):
    L_chan, a_chan, b_chan = tf.unstack(lab, axis=3)
    return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)

def lab_postprocess(lab):
    L_chan, a_chan, b_chan = tf.unstack(lab, axis=3)
    return tf.stack([L_chan/50-1, a_chan/110, b_chan/110], axis=3)

def lab_to_rgb(lab):
    #lab = check_image(lab)
    lab_pixels = tf.reshape(lab, [-1, 3])
    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    # convert to fxfyfz
    lab_to_fxfyfz = tf.constant([
        #   fx      fy        fz
        [1/116.0, 1/116.0,  1/116.0], # l
        [1/500.0,     0.0,      0.0], # a
        [    0.0,     0.0, -1/200.0], # b
    ])
    fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

    # convert to xyz
    epsilon = 6/29
    linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
    exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
    xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

    xyz_to_rgb = tf.constant([
        #     r           g          b
        [ 3.2404542, -0.9692660,  0.0556434], # x
        [-1.5371385,  1.8760108, -0.2040259], # y
        [-0.4985314,  0.0415560,  1.0572252], # z
    ])
    rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
    linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
    exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels) ** (1/2.4) * 1.055) - 0.055) * exponential_mask

    return tf.reshape(srgb_pixels, tf.shape(lab))


def rgb_to_lab(srgb):
    # based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
    srgb_pixels = tf.reshape(srgb, [-1, 3])
    linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
    exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
    
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055 * exponential_mask) ** 2.4)
    rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
    xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

    # normalize for D65 white point
    xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

    epsilon = 6/29
    linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
    exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + ((xyz_normalized_pixels * exponential_mask)** (1/3))

            # convert to lab
    fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
    lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

    return tf.reshape(lab_pixels, tf.shape(srgb))

#%%
def Discriminator():
    segmap0 = tf.keras.layers.Input(shape=[None, None, 7], name='input_image')
    im0 = tf.keras.layers.Input(shape=[None, None, 4], name='target_image')
    
    H = tf.expand_dims(im0[:,:,:,3], -1)
    top = H[:,1:,:,:]-H[:,:-1,:,:]
    top = tf.pad(top, [[0,0], [1,0], [0,0], [0,0]], mode='SYMMETRIC', name='top_pad')
    left = H[:,:,1:,:]-H[:,:,:-1,:]
    left = tf.pad(left, [[0,0], [0,0], [1,0], [0,0]], mode='SYMMETRIC', name='left_pad')
    im1 = tf.concat([im0, top, left], axis=3)
    
    D_logit = []
    for n in [128, 32, 8]:
        segmap = tf.image.resize(segmap0, [n, n], method=tf.image.ResizeMethod.BILINEAR)
        im = tf.image.resize(im1, [n, n], method=tf.image.ResizeMethod.BILINEAR)
        x = tf.concat([segmap, im], axis=3)

        channel = 128
        x = conv(x, channel, kernel=3, stride=2, use_bias=True)
        x = tf.nn.selu(x)
        #feature_loss.append(x)
        
        d = 2
        for i in range(d):
            #stride = 1 if i == d - 1 else 2
            stride = 2
            channel = min(channel * 2, 512)

            x = conv(x, channel, kernel=3, stride=stride, use_bias=True)
            x = tf.nn.selu(x)
            #feature_loss.append(x)
            
        x = conv(x, channels=1, kernel=1, stride=1, use_bias=True)
        #feature_loss.append(x)
        D_logit.append(x)

    return tf.keras.Model(inputs=[segmap0, im0], outputs=D_logit)
discriminator = Discriminator()


#%%
def discriminator_loss(real, fake):
    loss = []
    real_loss = 0
    fake_loss = 0
    for i in range(len(fake)):
        real_loss = -tf.reduce_mean(tf.minimum(real[i] - 1, 0.0))
        fake_loss = -tf.reduce_mean(tf.minimum(-fake[i] - 1, 0.0))
        loss.append(real_loss + fake_loss)
    return tf.reduce_mean(loss)


def generator_loss(fake):
    loss = []
    fake_loss = 0
    for i in range(len(fake)):
        #temp = tf.where(tf.math.is_nan(fake[i]), tf.zeros_like(fake[i]), fake[i])
        fake_loss = -tf.reduce_mean(fake[i])
        loss.append(fake_loss)
    return tf.reduce_mean(loss)


generator_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5, clipnorm=100)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, 0.5, clipnorm=100)

#%%
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    
    return tf.keras.Model([vgg.input], outputs)

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
style_extractor = vgg_layers(style_layers)
#style_outputs = style_extractor(style_image*255)

@tf.function
def vggloss(fake, real):
    fake = tf.keras.applications.vgg19.preprocess_input((fake+1)*127.5)
    fake = style_extractor(fake)
    
    real = tf.keras.applications.vgg19.preprocess_input((real+1)*127.5)
    real = style_extractor(real)
    
    loss = 0
    for i in range(len(fake)):
        loss += tf.reduce_mean(tf.abs(fake[i]-real[i]))
    return loss


#%%
checkpoint_dir = r'datasets\TERR\outH\ckpt'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 shadow_generator=shadow_generator)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5, keep_checkpoint_every_n_hours=2)
status = checkpoint.restore(manager.latest_checkpoint)


#%%
def generate_images(model, test_input, tar, htar, orig_inp, name):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    terrain = lab_to_rgb(lab_preprocess(tf.tanh(prediction[:,:,:,:3])))
    prediction = shadow_generator(prediction, training=True)
    prediction = tf.concat([lab_to_rgb(lab_preprocess(prediction[:,:,:,:3]))*2-1, tf.expand_dims(prediction[:,:,:,3], -1)], axis=3)
    
    hp = tf.expand_dims(tf.cast((prediction[:,:,:,3]+1)*127.5, tf.uint8), -1)
    prediction = tf.cast((prediction[:,:,:,:3]+1)*127.5, tf.uint8)
    terrain = tf.cast(terrain*255, tf.uint8)
    
    #tar = tf.cast((tar+1)*127.5, tf.uint8)
    #orig_inp = tf.cast((orig_inp+1)*127.5, tf.uint8)
    tar = tf.cast(tar, tf.uint8)
    orig_inp = tf.cast(orig_inp, tf.uint8)
    htar = tf.cast(tf.cast(htar, tf.float32)/257, tf.uint8)
    
    htar = tf.tile(htar, (1,1,1,3))
    hp = tf.tile(hp, (1,1,1,3))
    
    out1 = tf.concat([orig_inp, tar, htar], axis=2)
    out2 = tf.concat([terrain, prediction, hp], axis=2)
    out= tf.concat([out1, out2], axis=1)
      
    for i in range(out.shape.as_list()[0]):
        tf.io.write_file(os.path.join(r'C:\Users\tpapp\Desktop\GAN\datasets\TERR\outH',str(name)+'-'+str(i)+'.png'), tf.image.encode_png(out[i,:,:,:]))
        

@tf.function
def train_step(input_image, target, train_gen=True, train_dis=True):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_loss = 0
        disc_loss = 0
        
        gen_output = generator(input_image, training=True)
        #tf.print(tf.reduce_sum(tf.cast(tf.math.is_nan(gen_output), dtype=tf.float32)))
        gen_output1 = shadow_generator(gen_output, training=True)
        
        #target_lab = lab_postprocess(rgb_to_lab(target[:,:,:,:3]/2+0.5))
        vggs = vggloss(tf.stack([gen_output1[:,:,:,3], gen_output1[:,:,:,3], gen_output1[:,:,:,3]], 3), tf.stack([target[:,:,:,3], target[:,:,:,3], target[:,:,:,3]], 3))
        
        gen_output1 = tf.concat([lab_to_rgb(lab_preprocess(gen_output1[:,:,:,:3]))*2-1, tf.expand_dims(gen_output1[:,:,:,3], -1)], axis=3)
        
        flip = tf.concat([gen_output, input_image], axis=3)
        flip = tf.image.random_flip_left_right(flip)
        flip = tf.image.random_flip_up_down(flip)
        input_flip = flip[:,:,:,4:]
        gen_output2 = shadow_generator(flip[:,:,:,:4], training=True)
        gen_output2 = tf.concat([lab_to_rgb(lab_preprocess(gen_output2[:,:,:,:3]))*2-1, tf.expand_dims(gen_output2[:,:,:,3], -1)], axis=3)
        
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output1 = discriminator([input_image, gen_output1], training=True)
        disc_generated_output2 = discriminator([input_flip, gen_output2], training=True)
        
        vggl = vggloss(gen_output1[:,:,:,:3], target[:,:,:,:3]) + vggs
        #l1 = tf.reduce_mean(tf.abs(gen_output1-target))
        gen_loss = 0.001*vggl + generator_loss(disc_generated_output1) + generator_loss(disc_generated_output2) #+ l1
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output1) + discriminator_loss(disc_real_output, disc_generated_output2)
    
    generator_gradients = [0.0]
    if train_gen:
        gen_vars = generator.trainable_variables + shadow_generator.trainable_variables
        generator_gradients = gen_tape.gradient(gen_loss, gen_vars)
        if tf.math.is_inf(tf.linalg.global_norm(generator_gradients)):
            generator_gradients = [tf.clip_by_value(i, -1e16, 1e16) for i in generator_gradients]
            #generator_gradients = tf.clip_by_value(generator_gradients, -1e16, 1e16)
        generator_optimizer.apply_gradients(zip(generator_gradients, gen_vars))
    
    discriminator_gradients = [0.0]
    if train_dis:
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return disc_loss, gen_loss, vggl, tf.linalg.global_norm(discriminator_gradients), tf.linalg.global_norm(generator_gradients)#, (discriminator_gradients, generator_gradients)


def fit(train_ds, epochs, test_ds):
    L = []
    err = 0
    for epoch in range(epochs):
        start = time.time()
        
        # Train
        loss = []
        train_dis = True
        train_gen = True
        for input_image, target, ht in train_ds:
            target = tf.concat([target, ht], axis=3)
            temp = train_step(input_image, target, train_gen, train_dis)
            loss.append(list(temp) + [train_gen, train_dis])
            
            if epoch>0 and loss[-1][2] > 1000 and len(loss)>=10 and np.mean(np.array(loss[-10:])[:,2]) > 1000:
                err = 1
                break
            
            if loss[-1][0]>=2.0 and train_gen:
                train_gen = False
            elif loss[-1][0]<1.5 and (not train_gen):
                train_gen = True
            
        L.append(np.array(loss))
        for example_input, example_target, example_height, inp in test_ds.take(1):
            generate_images(generator, example_input, example_target, example_height, inp, epoch)
        
        if err==1:
            break
        
        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 1 == 0:
            manager.save()
        
        print('Epoch {} took {} min'.format(epoch, np.round((time.time()-start)/60, 2)))
        print(np.mean(L[-1][-100:,:], axis=0))
    return L

#%%
EPOCHS = 500
L = fit(train_dataset, EPOCHS, test_dataset)


#%%
# =============================================================================
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# for inp, tar in test_dataset.take(5):
#   generate_images(generator, inp, tar)
# =============================================================================
