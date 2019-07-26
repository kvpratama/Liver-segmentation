# -*- coding: utf-8 -*-
"""
Created on Thur 18 January 2018
@author: Mariëlle Jansen

Liver segmentation with 2D CNN

NOTE: First run loadData to load training and test data
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, Dropout, BatchNormalization, Activation, Concatenate, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras._impl.keras import backend as K
import loadData
import numpy as np
import SimpleITK as sitk

def dice_coef(y_true, y_pred, smooth=1e-5):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    THIS IS NOT DICE BUT ...
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    y_true = K.flatten(y_true[:,:,:,1]) 
    y_pred = K.flatten(y_pred[:,:,:,1]) 
    return 1-dice_coef(y_true, y_pred)

def diff_loss(y_true, y_pred):
    y_true = K.flatten(y_true) 
    y_pred = K.flatten(y_pred) 
    diff = K.sum(K.abs(y_true-y_pred))
    return diff

# define network
def build_network(Inputshape, num_class):
    input_img = Input(shape=Inputshape, name='Input_image')

    conv1 = Conv2D(32, (3,3), activation=None, use_bias=True,
                   kernel_initializer='glorot_uniform', name='conv1')(input_img)
    conv1 = BatchNormalization(axis=3, name='BN1')(conv1)
    conv1 = Activation(activation='relu', name='act_1')(conv1)
    
    conv2 = Conv2D(32, (3,3), activation=None, use_bias=True,
                   kernel_initializer='glorot_uniform', name='conv2')(conv1)
    conv2 = BatchNormalization(axis=3, name='BN2')(conv2)
    conv2 = Activation(activation='relu', name='act_2')(conv2)
    
    conv3 = Conv2D(32, (3,3), activation=None, dilation_rate=2, use_bias=True,
                   kernel_initializer='glorot_uniform', name='conv3')(conv2)
    conv3 = BatchNormalization(axis=3, name='BN3')(conv3)
    conv3 = Activation(activation='relu', name='act_3')(conv3)
    
    conv4 = Conv2D(32, (3,3), activation=None, dilation_rate=4, use_bias=True,
                   kernel_initializer='glorot_uniform', name='conv4')(conv3)
    conv4 = BatchNormalization(axis=3, name='BN4')(conv4)
    conv4 = Activation(activation='relu', name='act_4')(conv4)
    
    conv5 = Conv2D(32, (3,3), activation=None, dilation_rate=8, use_bias=True,
                   kernel_initializer='glorot_uniform', name='conv5')(conv4)
    conv5 = BatchNormalization(axis=3, name='BN5')(conv5)
    conv5 = Activation(activation='relu', name='act_5')(conv5)
    
    conv6 = Conv2D(32, (3,3), activation=None, dilation_rate=16, use_bias=True,
                   kernel_initializer='glorot_uniform', name='conv6')(conv5)
    conv6 = BatchNormalization(axis=3, name='BN6')(conv6)
    conv6 = Activation(activation='relu', name='act_6')(conv6)
    
    conv7 = Conv2D(32, (3,3), activation=None, dilation_rate=1, use_bias=True,
                   kernel_initializer='glorot_uniform', name='conv7')(conv6)   
    conv7 = BatchNormalization(axis=3, name='BN7')(conv7)
    out = Activation(activation='relu', name='act_7')(conv7)
    
    dropout = Dropout(0.5)(out)
    outFinal = Conv2D(2, (1,1), activation='softmax',
                      kernel_initializer='glorot_uniform', name='outFinal')(dropout)
    
    
    FinalModel = Model(inputs=input_img, outputs=outFinal)
    loss_fn = dice_coef_loss
    FinalModel.compile(optimizer=optimizers.Adam(lr=0.001, decay=0.0),
                       loss=loss_fn,
                       metrics=['accuracy'])
    return FinalModel

def iterate_in_mb_train(train_x, train_y):
    global batch_size
    
    while True:
        i = np.random.choice(train_x.shape[0], batch_size, replace=False)
        
        slices_input = train_x[i[0],:,:,:]
        slices_target = train_y[i[0],:,:,:]
        
        slices_input = slices_input[np.newaxis,:,:,:]
        slices_target = slices_target[np.newaxis,:,:,:]
        
        for p in range(1, int(batch_size)):
            p_input = train_x[i[p],:,:,:]
            p_target = train_y[i[p],:,:,:]
            
            p_input = p_input[np.newaxis,:,:,:]
            p_target = p_target[np.newaxis,:,:,:]
            
            slices_input = np.concatenate((slices_input, p_input), axis=0)
            slices_target = np.concatenate((slices_target, p_target), axis=0)
        
        mb_labels = np.zeros([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 2])
        neg_target = np.ones([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 1])
        neg_target = neg_target-slices_target
        mb_labels[:,:,:,0:1] = neg_target
        mb_labels[:,:,:,1:2] = slices_target
        
        
        yield slices_input, mb_labels

def iterate_in_mb_test(test_x, test_y):
    global batch_size
    
    while True:
        i = np.random.choice(test_x.shape[0], int(batch_size), replace=False)
        
        slices_input = test_x[i[0],:,:]
        slices_target = test_y[i[0],:,:]
        
        slices_input = slices_input[np.newaxis,:,:,:]
        slices_target = slices_target[np.newaxis,:,:,:]
        
        for p in range(1, int(batch_size)):
            p_input = train_x[i[p],:,:,:]
            p_target = train_y[i[p],:,:,:]
            
            p_input = p_input[np.newaxis,:,:,:]
            p_target = p_target[np.newaxis,:,:,:]
            
            slices_input = np.concatenate((slices_input, p_input), axis=0)
            slices_target = np.concatenate((slices_target, p_target), axis=0)
        
        mb_labels = np.zeros([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 2])
        neg_target = np.ones([slices_target.shape[0], slices_target.shape[1], slices_target.shape[2], 1])
        neg_target = neg_target-slices_target
        mb_labels[:,:,:,0:1] = neg_target
        mb_labels[:,:,:,1:2] = slices_target
        
        
        yield slices_input, mb_labels

# Test and save segmentation
def imsave(fname, arr):
    #arr = np.swapaxes(arr, 0, 2)
    sitk_img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(sitk_img, fname)
  
n_classes = 2
batch_size = 1
train_x, train_y, test_x, test_y = loadData.load_data()

## Training step
FinalModel = build_network(Inputshape=(322, 322, 6), num_class=2)
 
tbCallback = TensorBoard(log_dir='/input/logs/2DFCCN_6channels_input_run2',
                         histogram_freq=0, write_graph=True,
                         write_images=True)
n_epochs = 500000
FinalModel.fit_generator(iterate_in_mb_train(train_x, train_y), 1,
                         epochs=n_epochs, callbacks=[tbCallback], verbose=0,
                         validation_data=iterate_in_mb_test(test_x, test_y),
                         validation_steps=1) 

FinalModel.save_weights('/input/LiverFCNN_2D_6channels_.h5', overwrite=True)

LiverSegmentation = build_network(Inputshape=(322, 322, 6), num_class=2)
LiverSegmentation.load_weights('/input/LiverFCNN_2D_6channels_.h5', by_name=False)

## Testing step
for t in range(1,4):
    test_50 = test_x[(t-1)*100,:,:,:]
    test_50 = test_50[np.newaxis, :, :, :]
    predict_50 = LiverSegmentation.predict([test_50])
    predict_50 = np.reshape(predict_50, [256,256,2])
    prediction = predict_50[:,:,1]
    prediction = prediction[np.newaxis,:,:]
    
    for slice in range(1+((t-1)*100),(t)*100):
        test_50 = test_x[slice,:,:,:]
        test_50 = test_50[np.newaxis, :, :, :]
        predict_50 = LiverSegmentation.predict([test_50])
        predict_50 = np.reshape(predict_50, [256,256,2])
        predict_50 = predict_50[:,:,1]*100
        prediction = np.concatenate((prediction, predict_50[np.newaxis,:,:]), axis=0)
    
    imsave('/input/output/FCNN_2D_6channels/result_test_'+str(t)+'_.nii', prediction)



        
    
