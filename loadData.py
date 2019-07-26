# -*- coding: utf-8 -*-
"""
Created on Thur 18 January 2018

@author: Mariëlle Jansen

Load data for liver segmentation - with 6 input images
"""
# The target segmentations do not necessarily correspond to the loaded data. This
# should be coupled before using this again.
def load_data():
    import numpy as np
    import SimpleITK as sitk
    from os import path
    
    
    
    def read_image(filename):
        image = sitk.ReadImage(filename)  # Use ITK to read the image
        image = sitk.GetArrayFromImage(image)  # Turn ITK image object into a numpy array
        return image
    
#    train_i = np.asarray([2, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#    test_i = np.asarray([1, 3, 4]) # exclude , 11
    
    def unit_load(nameImage, arg):
        basedir = r'/input/Pre-processed-6channels'
        #basedir= r'H:\Docker\Pre-processed-6channels'
        # Load train data; find body mask and normalize
        train = read_image(path.join(basedir, str(arg[0])+str(nameImage)))
        train = train.astype('float32')
        bg = train[0,0,0]
        # Zero padding of 33 in axis 1 and 2
        train_zeros = np.ones([train.shape[0], train.shape[1]+66, train.shape[2]+66])*bg
        train_zeros[:,33:-33,33:-33] = train
        train = train_zeros
        train /= 100 # normalization
        y_train = read_image(path.join(basedir, str(arg[0])+'_targetMask.nii'))
		
        for i in range(1,len(arg)):
            train_1 = read_image(path.join(basedir, str(arg[i])+str(nameImage)))
            train_1 = train_1.astype('float32')
            bg = train_1[0,0,0]
            y_train_1 = read_image(path.join(basedir, str(arg[i])+'_targetMask.nii'))
            # Zero padding of 33 in axis 1 and 2
            train_zeros = np.ones([train_1.shape[0], train_1.shape[1]+66, train_1.shape[2]+66])*bg
            train_zeros[:,33:-33,33:-33] = train_1
            train_1 = train_zeros
            train_1 /= 100 # normalization
            
            train = np.concatenate((train, train_1), axis=0)
            y_train = np.concatenate((y_train, y_train_1), axis=0)
        
        return train, y_train
        
    def load_train():
        train_i = np.asarray([2, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    
        train_x, train_y = unit_load(nameImage='_inputImage_1.nii', arg=train_i)
        train_x = train_x[:,:,:, np.newaxis]
        train_y = train_y[:,:,:, np.newaxis]
		
        train, _ = unit_load(nameImage='_inputImage_2.nii', arg=train_i)
        train_x = np.concatenate((train_x, train[:,:,:, np.newaxis]), axis=3)
        
        train, _ = unit_load(nameImage='_inputImage_3.nii', arg=train_i)
        train_x = np.concatenate((train_x, train[:,:,:, np.newaxis]), axis=3)
        
        train, _ = unit_load(nameImage='_inputImage_4.nii', arg=train_i)
        train_x = np.concatenate((train_x, train[:,:,:, np.newaxis]), axis=3)
        
        train, _ = unit_load(nameImage='_inputImage_5.nii', arg=train_i)
        train_x = np.concatenate((train_x, train[:,:,:, np.newaxis]), axis=3)
        
        train, _ = unit_load(nameImage='_inputImage_6.nii', arg=train_i)
        train_x = np.concatenate((train_x, train[:,:,:, np.newaxis]), axis=3)
        
        
        print('Loading train set complete.')
        return train_x, train_y
    
    def load_test():
        test_i = np.asarray([1, 3, 4]) # exclude , 11
        
        test_x, test_y = unit_load(nameImage='_inputImage_1.nii', arg=test_i)
        test_x = test_x[:,:,:, np.newaxis]
        test_y = test_y[:,:,:, np.newaxis]
        
        test, _ = unit_load(nameImage='_inputImage_2.nii', arg=test_i)
        test_x = np.concatenate((test_x, test[:,:,:, np.newaxis]), axis=3)
        
        test, _ = unit_load(nameImage='_inputImage_3.nii', arg=test_i)
        test_x = np.concatenate((test_x, test[:,:,:, np.newaxis]), axis=3)
        
        test, _ = unit_load(nameImage='_inputImage_4.nii', arg=test_i)
        test_x = np.concatenate((test_x, test[:,:,:, np.newaxis]), axis=3)
        
        test, _ = unit_load(nameImage='_inputImage_5.nii', arg=test_i)
        test_x = np.concatenate((test_x, test[:,:,:, np.newaxis]), axis=3)
        
        test, _ = unit_load(nameImage='_inputImage_6.nii', arg=test_i)
        test_x = np.concatenate((test_x, test[:,:,:, np.newaxis]), axis=3)
        
        print('Loading test set complete.')
        return test_x, test_y
    
    
    train_x, train_y = load_train()
    test_x, test_y = load_test()
    
    return train_x, train_y, test_x, test_y
