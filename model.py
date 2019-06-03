# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:03:57 2016

@author: Chase
"""

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, concatenate, SpatialDropout2D, add, Reshape
from keras.layers import Convolution2D, AveragePooling2D, UpSampling2D, Flatten, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras import metrics
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
from keras.losses import mse, binary_crossentropy

from keras.layers import Lambda, Input, Dense
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_distance_loss(y_true, y_pred, smooth=10):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def vae_loss(inputs,outputs):

    reconstruction_loss = jaccard_distance_loss(inputs, outputs)
    reconstruction_loss = K.mean(reconstruction_loss)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    kl_loss = kl_loss/256**2
    return K.mean(reconstruction_loss + kl_loss)



def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X &amp; Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    return (intersection + smooth) / ( union + smooth)




def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)

def unet1024(img_rows,img_cols):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = 'layer1.1')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = 'layer1.2')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name = 'layer1.3')(conv1)
    
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'layer2.1')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'layer2.2')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name = 'layer2.3')(conv2)
    
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'layer3.1')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'layer3.2')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name = 'layer3.3')(conv3)
    
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'layer4.1')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'layer4.2')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name = 'layer4.3')(conv4)
    
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'layer5.1')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'layer5.2')(conv5)
    conv5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name = 'layer5.3')(conv5)
    
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name = 'layer6.1')(pool5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name = 'layer6.2')(conv6)
    conv6 = BatchNormalization()(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name = 'layer6.3')(conv6)
    
    conv7 = Conv2D(1024, (3, 3), activation='relu', padding='same', name = 'layer7.1')(pool6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(1024, (3, 3), activation='relu', padding='same', name = 'layer7.2')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    up8 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name = 'layer8.0')(conv7), conv6], axis=3, name = 'layer8.01')
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same', name = 'layer8.1')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same', name = 'layer8.2')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    up9 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name = 'layer9.0')(conv8), conv5], axis=3, name = 'layer9.01')
    conv9 = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'layer9.1')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'layer9.2')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    up10 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name = 'layer10.0')(conv9), conv4], axis=3, name = 'layer10.01')
    conv10 = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'layer10.1')(up10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'layer10.2')(conv10)
    conv10 = BatchNormalization()(conv10)
    
    up11 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name = 'layer11.0')(conv10), conv3], axis=3, name = 'layer11.01')
    conv11 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'layer11.1')(up11)
    conv11 = BatchNormalization()(conv11)
    conv11 = Conv2D(128, (3, 3), activation='relu', padding='same', name = 'layer11.2')(conv11)
    conv11 = BatchNormalization()(conv11)
    
    up12 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name = 'layer12.0')(conv11), conv2], axis=3, name = 'layer12.01')
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'layer12.1')(up12)
    conv12 = BatchNormalization()(conv12)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', name = 'layer12.2')(conv12)
    conv12 = BatchNormalization()(conv12)
    
    up13 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name = 'layer13.0')(conv12), conv1], axis=3, name = 'layer13.01')
    conv13 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'layer13.1')(up13)
    conv13 = BatchNormalization()(conv13)
    conv13 = Conv2D(32, (3, 3), activation='relu', padding='same', name = 'layer13.2')(conv13)
    conv13 = BatchNormalization()(conv13)
    
    conv14 = Conv2D(1, (1, 1), activation='sigmoid')(conv13)
    # conv14 = BatchNormalization()(conv14)
    
    model = Model(inputs=[inputs], outputs=[conv14])
    model.compile(optimizer = Adam(lr = 1e-4,decay=1e-6), loss = 'binary_crossentropy', metrics = [jaccard_distance_loss,iou_coef,dice_coef])
    return model
    
def get_cnn_model(img_rows=128, img_cols=128,nchannels=1,pretrained_weights = None):
    input_size=(None,None,nchannels)
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=4)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=4)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=4)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=4)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=4)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=4)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    concatenate6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concatenate6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    concatenate7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concatenate7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    concatenate8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concatenate8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    concatenate9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concatenate9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-5,decay=1e-6), loss = 'binary_crossentropy', metrics = [jaccard_distance_loss,iou_coef,dice_coef])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model



def get_iterative_model(img_rows, img_cols):
    # the heatmap from CNN
    input1 = Input(shape=(None, None,1))
    
    # the image heatmap was generated from (change 1 to 3 for color images)
    input2 = Input(shape=(None, None,1))
    
    # these are shared layer declarations
    convh1 = Convolution2D(32, 3, activation='relu', border_mode='same')
    convh2 = Convolution2D(64, 3, activation='relu', border_mode='same')
    
    convi1 = Convolution2D(32, 3, activation='relu', border_mode='same')
    
    convmlp1 = Convolution2D(32, 1, activation='relu')
    convmlp2 = Convolution2D(32,1, activation='relu')
    convmlp3 = Convolution2D(1, 1)
    
    # now the actual model
    upperconv1 = convh1(input1)
    lowerconv1 = convi1(input2)
    
    concat1 = concatenate([upperconv1, lowerconv1],axis=3)
    upperconv2 = convh2(concat1)
    
    mlp11 = convmlp1(upperconv2)
    mlp12 = convmlp2(mlp11)
    mlp13 = convmlp3(mlp12)
    
    add1 = add([input1, mlp13])
    
    
    upperconv2 = convh1(add1)
    lowerconv2 = convi1(input2)
    
    concat2 = concatenate([upperconv2, lowerconv2], axis=3)
    upperconv3 = convh2(concat2)
    
    mlp21 = convmlp1(upperconv3)
    mlp22 = convmlp2(mlp21)
    mlp23 = convmlp3(mlp22)
    
    add2 = add([add1, mlp23])
    
    
    upperconv3 = convh1(add2)
    lowerconv3 = convi1(input2)
    
    concat3 = concatenate([upperconv3, lowerconv3], axis=3)
    upperconv4 = convh2(concat3)
    
    mlp31 = convmlp1(upperconv4)
    mlp32 = convmlp2(mlp31)
    mlp33 = convmlp3(mlp32)
    
    add3 = add([add2, mlp33])
    
    
    upperconv4 = convh1(add3)
    lowerconv4 = convi1(input2)
    
    concat4 = concatenate([upperconv4, lowerconv4],  axis=3)
    upperconv5 = convh2(concat4)
    
    mlp41 = convmlp1(upperconv5)
    mlp42 = convmlp2(mlp41)
    mlp43 = convmlp3(mlp42)
    
    add4 = add([add3, mlp43])
    
    
    upperconv5 = convh1(add4)
    lowerconv5 = convi1(input2)
    
    concat5 = concatenate([upperconv5, lowerconv5], axis=3)
    upperconv6 = convh2(concat5)
    
    mlp51 = convmlp1(upperconv6)
    mlp52 = convmlp2(mlp51)
    mlp53 = convmlp3(mlp52)
    
    add5 = add([add4, mlp53])
    
    
    model = Model(input=[input1, input2], output=add5)
    model.compile(optimizer = Adam(lr = 1e-4,decay=1e-6), loss = 'binary_crossentropy', metrics = [jaccard_distance_loss,iou_coef,dice_coef])
    
    return model

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_conv_vae_model(rows=128,cols=128,nchannels=2,filters=1,num_conv=3):
    input_shape = (rows,cols,nchannels)
    intermediate_dim = 512
    
    latent_dim = 4
    
    
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    conv_1 = Conv2D(nchannels,
                    kernel_size=(2, 2),
                    padding='same', activation='relu')(inputs)
    conv_2 = Conv2D(filters,
                    kernel_size=(2, 2),
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_2)
    conv_4 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_3)
    flat = Flatten()(conv_4)
    x = Dense(intermediate_dim, activation='relu')(flat)
    global z_mean
    z_mean = Dense(latent_dim, name='z_mean')(x)
    global z_log_var
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    output_shape = (rows // 2, cols // 2, filters)
    decoder_upsample = Dense(filters * rows//2 *cols//2, activation='relu')
    decoder_reshape = Reshape(output_shape)
    decoder_deconv_1 = Conv2DTranspose(filters,
    kernel_size=num_conv,
    padding='same',
    strides=1,
    activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters,
    kernel_size=num_conv,
    padding='same',
    strides=1,
    activation='relu')
    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding='valid',
    activation='relu')
    decoder_mean_squash = Conv2D(1,
    kernel_size=2,
    padding='valid',
    activation='sigmoid')

    up_decoded = decoder_upsample(x)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    outputs = decoder_mean_squash(x_decoded_relu)

    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    
    models = (encoder, decoder)
    
    
    vae.compile(optimizer = Adam(lr = 1e-4,decay=1e-6),loss=vae_loss)
    return vae, models

def get_vae_model(rows=128,cols=128,nchannels=2):
    input_shape = (rows,cols,nchannels )
    intermediate_dim = cols*2
    original_dim = rows*cols
    latent_dim = 4
    
    
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    global z_mean
    z_mean = Dense(latent_dim, name='z_mean')(x)
    global z_log_var
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    
    models = (encoder, decoder)
    
    

    vae.compile(optimizer = Adam(lr = 1e-4,decay=1e-6),loss=vae_loss,metrics = [jaccard_distance_loss,iou_coef,dice_coef])
    return vae, models