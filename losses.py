import numpy as np
import keras.backend as K


def dice(y_true, y_pred):
    #computes the dice score on two tensors

    sum_p=K.sum(y_pred,axis=0)
    sum_r=K.sum(y_true,axis=0)
    sum_pr=K.sum(y_true * y_pred,axis=0)
    dice_numerator =2*sum_pr
    dice_denominator =sum_r+sum_p
    dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
    return dice_score


def dice_whole_metric(y_true, y_pred):
    #computes the dice for the whole tumor

    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    y_whole=K.sum(y_true_f[:,1:],axis=1)
    p_whole=K.sum(y_pred_f[:,1:],axis=1)
    dice_whole=dice(y_whole,p_whole)
    return dice_whole

def dice_en_metric(y_true, y_pred):
    #computes the dice for the enhancing region

    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    y_enh=y_true_f[:,-1]
    p_enh=y_pred_f[:,-1]
    dice_en=dice(y_enh,p_enh)
    return dice_en

def dice_core_metric(y_true, y_pred):
    ##computes the dice for the core region

    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    
    #workaround for tf
    #y_core=K.sum(tf.gather(y_true_f, [1,3],axis =1),axis=1)
    #p_core=K.sum(tf.gather(y_pred_f, [1,3],axis =1),axis=1)
    
    y_core=K.sum(y_true_f[:,[1,3]],axis=1)
    p_core=K.sum(y_pred_f[:,[1,3]],axis=1)
    dice_core=dice(y_core,p_core)
    return dice_core



def weighted_log_loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # weights are assigned in this order : normal,necrotic,edema,enhancing 
    weights=np.array([1,5,2,4])
    weights = K.variable(weights)
    loss = y_true * K.log(y_pred) * weights
    loss = K.mean(-K.sum(loss, -1))
    return loss

def gen_dice_loss(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss and weighted cross entropy
    '''

    #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    sum_p=K.sum(y_pred_f,axis=-2)
    sum_r=K.sum(y_true_f,axis=-2)
    sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
    weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
    generalised_dice_numerator =2*K.sum(weights*sum_pr)
    generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
    generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
    GDL=1-generalised_dice_score
    del sum_p,sum_r,sum_pr,weights

    return GDL+weighted_log_loss(y_true,y_pred)
