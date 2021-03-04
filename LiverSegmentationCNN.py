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
from tensorflow.python.keras import backend as K
import loadData
import numpy as np
import SimpleITK as sitk
from PIL import Image
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import pdb

def dice_coef(y_true, y_pred, smooth=1e-5):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    THIS IS NOT DICE BUT ...
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_test(y_true, y_pred, smooth=1e-5):
    intersection = np.sum(np.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (np.sum(np.square(y_true)) + np.sum(np.square(y_pred)) + smooth)

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

def iterate_in_mb_train(train_x):
    global batch_size
    
    while True:
        selected = np.random.choice(len(train_x), 1, replace=False)
        slices_input, mb_labels = loadData.loadtrain(train_x[selected[0]], batch_size)
        
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


def save_prediction(arr, path):
    fileobj = open(path, mode='wb')
    off = np.array(arr >= 0.5, dtype=np.int8)
    off.tofile(fileobj)
    fileobj.close()


n_classes = 2
batch_size = 8

basedir = r'Z:\data4\livertumormri\train20210210_seg_green'
ckptdir = 'D:\weights\livertumormri\\20200304_tf'
train_paths = glob.glob(basedir + '\\train3d\data\*.nii')
test_paths = glob.glob(basedir + '\\val3d\data\*.nii')

## Training step
FinalModel = build_network(Inputshape=(322, 322, 1), num_class=2)
 
# tbCallback = TensorBoard(log_dir=ckptdir,
#                          histogram_freq=0, write_graph=True,
#                          write_images=True)
n_epochs = 1
n_iter = 10
for epoch in range(n_epochs):
    FinalModel.fit_generator(iterate_in_mb_train(train_paths), 1,
                             epochs=n_iter, verbose=1,
                             validation_data=iterate_in_mb_train(test_paths),
                             validation_steps=1)

    FinalModel.save_weights(ckptdir + f'\\LiverFCNN_2D_6channels_{epoch}.h5', overwrite=True)

print('Testing Step')
ckpts = glob.glob(ckptdir + '\\*.h5')
os.makedirs(os.path.dirname(ckptdir + "/save/"), exist_ok=True)
for i, ckpt in enumerate(ckpts):
    LiverSegmentation = build_network(Inputshape=(322, 322, 1), num_class=2)
    LiverSegmentation.load_weights(ckpt, by_name=False)
    print('Load model: ', ckpt)
    dice_list = []

    for test_path in test_paths:
        test_x = loadData.read_image(test_path)
        test_x = test_x.astype('float32')
        depth, height, width = test_x.shape
        bg = test_x[0, 0, 0]
        test_y = loadData.read_mask(
            os.path.dirname(os.path.dirname(test_path)) + '\\mask\\' + os.path.basename(test_path)[:-4] + '_gt1.raw', depth,
            height, width)

        prediction_3d = np.zeros((depth, height, width))
        for j in range(depth):
            input_x = np.zeros((1, 322, 322, 1))
            slice_data = test_x[j]

            input_img = Image.fromarray(slice_data)
            input_img = input_img.resize((256, 256), Image.LANCZOS)
            slice_data = np.array(input_img)

            train_zeros = np.ones([slice_data.shape[0] + 66, slice_data.shape[1] + 66]) * bg
            train_zeros[33:-33, 33:-33] = slice_data
            train_zeros /= 100  # normalization

            input_x[0, :, :, 0] = train_zeros
            prediction = LiverSegmentation.predict([input_x])
            try:
                pred_img = Image.fromarray(prediction[0, :, :, 1])
                pred_img = pred_img.resize((width, height), Image.LANCZOS)
                prediction_3d[j] = np.array(pred_img)
            except:
                pdb.set_trace()

        dice = dice_coef_test(test_y.astype('float32'), prediction_3d.astype('float32'))
        print('Test: ', test_path, dice)
        dice_list.append((test_path, dice))
        # save_prediction(prediction_3d, ckptdir + "/save/" + str(i + 1) + "_" + os.path.basename(test_path)[:-4] + '.raw')

    dice_df = pd.DataFrame(dice_list, columns=['filepath', 'dice'])
    dice_df.to_csv(f"{ckptdir}/save/dice_{i}.csv")
