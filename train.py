import numpy as np
import random
import json
from glob import glob
from keras.models import model_from_json,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint,Callback,LearningRateScheduler
import keras.backend as K
from model import Unet_model
from losses import *
#from keras.utils.visualize_util import plot



class SGDLearningRateTracker(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.get_value(optimizer.lr)
        decay = K.get_value(optimizer.decay)
        lr=lr/10
        decay=decay*10
        K.set_value(optimizer.lr, lr)
        K.set_value(optimizer.decay, decay)
        print('LR changed to:',lr)
        print('Decay changed to:',decay)



class Training(object):
    

    def __init__(self, batch_size,nb_epoch,load_model_resume_training=None):

        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

        #loading model from path to resume previous training without recompiling the whole model
        if load_model_resume_training is not None:
            self.model =load_model(load_model_resume_training,custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
            print("pre-trained model loaded!")
        else:
            unet =Unet_model(img_shape=(128,128,4))
            self.model=unet.model
            print("U-net CNN compiled!")

                    
    def fit_unet(self,X33_train,Y_train,X_patches_valid=None,Y_labels_valid=None):

        train_generator=self.img_msk_gen(X33_train,Y_train,9999)
        checkpointer = ModelCheckpoint(filepath='brain_segmentation/ResUnet.{epoch:02d}_{val_loss:.3f}.hdf5', verbose=1)
        self.model.fit_generator(train_generator,steps_per_epoch=len(X33_train)//self.batch_size,epochs=self.nb_epoch, validation_data=(X_patches_valid,Y_labels_valid),verbose=1, callbacks = [checkpointer,SGDLearningRateTracker()])
        #self.model.fit(X33_train,Y_train, epochs=self.nb_epoch,batch_size=self.batch_size,validation_data=(X_patches_valid,Y_labels_valid),verbose=1, callbacks = [checkpointer,SGDLearningRateTracker()])

    def img_msk_gen(self,X33_train,Y_train,seed):

        '''
        a custom generator that performs data augmentation on both patches and their corresponding targets (masks)
        '''
        datagen = ImageDataGenerator(horizontal_flip=True,data_format="channels_last")
        datagen_msk = ImageDataGenerator(horizontal_flip=True,data_format="channels_last")
        image_generator = datagen.flow(X33_train,batch_size=4,seed=seed)
        y_generator = datagen_msk.flow(Y_train,batch_size=4,seed=seed)
        while True:
            yield(image_generator.next(), y_generator.next())


    def save_model(self, model_name):
        '''
        INPUT string 'model_name': path where to save model and weights, without extension
        Saves current model as json and weights as h5df file
        '''

        model_tosave = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        json_string = self.model.to_json()
        self.model.save_weights(weights)
        with open(model_tosave, 'w') as f:
            json.dump(json_string, f)
        print ('Model saved.')

    def load_model(self, model_name):
        '''
        Load a model
        INPUT  (1) string 'model_name': filepath to model and weights, not including extension
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        '''
        print ('Loading model {}'.format(model_name))
        model_toload = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        with open(model_toload) as f:
            m = next(f)
        model_comp = model_from_json(json.loads(m))
        model_comp.load_weights(weights)
        print ('Model loaded.')
        self.model = model_comp
        return model_comp



if __name__ == "__main__":
    #set arguments

    #reload already trained model to resume training
    model_to_load="Models/ResUnet.04_0.646.hdf5" 
    #save=None

    #compile the model
    brain_seg = Training(batch_size=4,nb_epoch=3,load_model_resume_training=model_to_load)

    print("number of trainabale parameters:",brain_seg.model.count_params())
    #print(brain_seg.model.summary())
    #plot(brain_seg.model, to_file='model_architecture.png', show_shapes=True)

    #load data from disk
    Y_labels=np.load("y_training.npy").astype(np.uint8)
    X_patches=np.load("x_training.npy").astype(np.float32)
    Y_labels_valid=np.load("y_valid.npy").astype(np.uint8)
    X_patches_valid=np.load("x_valid.npy").astype(np.float32)
    print("loading patches done\n")

    # fit model
    brain_seg.fit_unet(X_patches,Y_labels,X_patches_valid,Y_labels_valid)#*

    #if save is not None:
    #    brain_seg.save_model('models/' + save)




