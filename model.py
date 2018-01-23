
import numpy as np
from keras.models import Model,load_model
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout,GaussianNoise, Input,Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import  Conv2DTranspose,UpSampling2D,concatenate,add
from keras.optimizers import SGD
import keras.backend as K
from losses import *

K.set_image_data_format("channels_last")

 #u-net model
class Unet_model(object):
    
    def __init__(self,img_shape,load_model_weights=None):
        self.img_shape=img_shape
        self.load_model_weights=load_model_weights
        self.model =self.compile_unet()
        
    
    def compile_unet(self):
        """
        compile the U-net model
        """
        i = Input(shape=self.img_shape)
        #add gaussian noise to the first layer to combat overfitting
        i_=GaussianNoise(0.01)(i)

        i_ = Conv2D(64, 2, padding='same',data_format = 'channels_last')(i_)
        out=self.unet(inputs=i_)
        model = Model(input=i, output=out)

        sgd = SGD(lr=0.08, momentum=0.9, decay=5e-6, nesterov=False)
        model.compile(loss=gen_dice_loss, optimizer=sgd, metrics=[dice_whole_metric,dice_core_metric,dice_en_metric])
        #load weights if set for prediction
        if self.load_model_weights is not None:
            model.load_weights(self.load_model_weights)
        return model


    def unet(self,inputs, nb_classes=4, start_ch=64, depth=3, inc_rate=2. ,activation='relu', dropout=0.0, batchnorm=True, upconv=True,format_='channels_last'):
        """
        the actual u-net architecture
        """
        o = self.level_block(inputs,start_ch, depth, inc_rate,activation, dropout, batchnorm, upconv,format_)
        o = BatchNormalization()(o) 
        #o =  Activation('relu')(o)
        o=PReLU(shared_axes=[1, 2])(o)
        o = Conv2D(nb_classes, 1, padding='same',data_format = format_)(o)
        o = Activation('softmax')(o)
        return o



    def level_block(self,m, dim, depth, inc, acti, do, bn, up,format_="channels_last"):
        if depth > 0:
            n = self.res_block_enc(m,0.0,dim,acti, bn,format_)
            #using strided 2D conv for donwsampling
            m = Conv2D(int(inc*dim), 2,strides=2, padding='same',data_format = format_)(n)
            m = self.level_block(m,int(inc*dim), depth-1, inc, acti, do, bn, up )
            if up:
                m = UpSampling2D(size=(2, 2),data_format = format_)(m)
                m = Conv2D(dim, 2, padding='same',data_format = format_)(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2,padding='same',data_format = format_)(m)
            n=concatenate([n,m])
            #the decoding path
            m = self.res_block_dec(n, 0.0,dim, acti, bn, format_)
        else:
            m = self.res_block_enc(m, 0.0,dim, acti, bn, format_)
        return m

  
   
    def res_block_enc(self,m, drpout,dim,acti, bn,format_="channels_last"):
        
        """
        the encoding unit which a residual block
        """
        n = BatchNormalization()(m) if bn else n
        #n=  Activation(acti)(n)
        n=PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same',data_format = format_)(n)
                
        n = BatchNormalization()(n) if bn else n
        #n=  Activation(acti)(n)
        n=PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same',data_format =format_ )(n)

        n=add([m,n]) 
        
        return  n 



    def res_block_dec(self,m, drpout,dim,acti, bn,format_="channels_last"):

        """
        the decoding unit which a residual block
        """
         
        n = BatchNormalization()(m) if bn else n
        #n=  Activation(acti)(n)
        n=PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same',data_format = format_)(n)

        n = BatchNormalization()(n) if bn else n
        #n=  Activation(acti)(n)
        n=PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same',data_format =format_ )(n)
        
        Save = Conv2D(dim, 1, padding='same',data_format = format_,use_bias=False)(m) 
        n=add([Save,n]) 
        
        return  n   



    
