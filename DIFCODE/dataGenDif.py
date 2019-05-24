# -*- coding: utf-8 -*-

import glob
import os
#import cv2

from keras.preprocessing import image


import numpy as np
#from multiprocessing import Pool


patch_size, stride = 40, 10
aug_times = 1
#scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):

    # read image

    img = image.load_img(file_name,grayscale=True)
    x = image.img_to_array(img)
    h, w, dummy = x.shape
    patches = []

    # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            thisx = x[i:i+patch_size, j:j+patch_size]
            #patches.append(x)
            # data aug
            for k in range(0, aug_times):
                thisx_aug = data_aug(thisx, mode=np.random.randint(0,8))
                #thisx_aug = data_aug(thisx, mode=0)
                patches.append(thisx_aug)

    return patches

def datagenerator(origDir, noiseDir, denoiseDir0, denoiseDir1):
    verbose = True
    file_list = glob.glob(origDir+'*.png')  # get name list of all .png files
    # initrialize
    dataO = []
    dataN = []
    data0 = []
    data1 = []# generate patches
    for i in range(len(file_list)):
    #for i in range(10):
        head, tail = os.path.split(file_list[i])
        orig = file_list[i]
        noise = noiseDir + tail
        denoise0 = denoiseDir0 + tail
        denoise1 = denoiseDir1 + tail
        thisseed = np.random.randint(0, 10000, 1)
        np.random.seed(thisseed)
        patchO = gen_patches(orig)
        np.random.seed(thisseed)
        patchN = gen_patches(noise)
        np.random.seed(thisseed)
        patch0 = gen_patches(denoise0)
        np.random.seed(thisseed)
        patch1 = gen_patches(denoise1)

        dataO.append(patchO)
        dataN.append(patchN)
        data0.append(patch0)
        data1.append(patch1)
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done ^_^')

    dataO = postProc(dataO)
    dataN = postProc(dataN)
    data0 = postProc(data0)
    data1 = postProc(data1)

    print('^_^-training data finished-^_^')
    return dataO, dataN, data0, data1

def postProc(data):
    data = np.array(data, dtype='uint8')
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],1))
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis = 0)
    return data

if __name__ == '__main__':
    import pudb; pu.db
    baseDir = '/home/cosc/csprh/linux/code/den2/data/'
    origDir = baseDir + 'Origs/Train/'
    noisyDir = baseDir + 'AWGN25/Train/'
    noiseDir = noisyDir +  'NoiseOrigs/';
    denoiseDir0 = noisyDir + 'Denoise0/';
    denoiseDir1 = noisyDir + 'Denoise1/';
    dataO, dataN, data0, data1 = datagenerator(origDir, noiseDir, denoiseDir0, denoiseDir1)


