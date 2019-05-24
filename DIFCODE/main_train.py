# Multiple Inputs
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, Add, Average
import numpy as np
#import tensorflow as tf
from keras.models import *
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Lambda,Subtract

import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import dataGenDif as dgd
import keras.backend as K

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='difCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='/home/cosc/csprh/linux/code/den2/data/', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=300, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=5, type=int, help='save model at every x epoches')
args = parser.parse_args()


save_dir = os.path.join('models',args.model+'_'+'sigma'+str(args.sigma))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def difCNN(depth,filters=64):

    # first input model
    inpt3 = Input(shape=(None,None,1))
    inpt1 = Input(shape=(None,None,1))
    # 1st layer, Conv+relu
    x1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt1)
    x1 = Activation('relu')(x1)
    # 15 layers, Conv+BN+relu
    for i in range(depth):
       x1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x1)
       x1 = BatchNormalization(axis=-1, epsilon=1e-3)(x1)
       x1 = Activation('relu')(x1)
    # last layer, Conv
    x1 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x1)

    # first input model
    inpt2 = Input(shape=(None,None,1))
    # 1st layer, Conv+relu
    x2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt2)
    x2 = Activation('relu')(x2)
    # 15 layers, Conv+BN+relu
    for ii in range(depth):
       x2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x2)
       x2 = BatchNormalization(axis=-1, epsilon=1e-3)(x2)
       x2 = Activation('relu')(x2)
    # last layer, Conv
    x2 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x2)
    sub = Average()([x1, x2])
    #sub = Subtract()([inpt3, added])   # input - noise

    model = Model(inputs=[inpt1, inpt2, inpt3], outputs=sub)
    return model

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_new_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=30:
        lr = initial_lr
    elif epoch<=60:
        lr = initial_lr/10
    elif epoch<=80:
        lr = initial_lr/20
    else:
        lr = initial_lr/20
    log('current learning rate is %2.8f' %lr)
    return lr

def train_datagen(epoch_iter=2000,epoch_num=5,batch_size=128,baseDir=args.train_data):
    while(True):
        n_count = 0
        if n_count == 0:
            origDir = baseDir + 'Origs/Train/'
            noisyDir = baseDir + 'AWGN25/Train/' #should adjust for different sigmas
            noiseDir = noisyDir +  'NoiseOrigs/'
            denoiseDir0 = noisyDir + 'Denoise0/'
            denoiseDir1 = noisyDir + 'Denoise1/'
            #print(n_count)
            dataO, dataN, data0, data1 = dgd.datagenerator(origDir, noiseDir, denoiseDir0, denoiseDir1)
            #assert len(dataO)%args.batch_size ==0, \
            #assert len(dataN)%args.batch_size ==0, \
            #assert len(data0)%args.batch_size ==0, \
            #assert len(data1)%args.batch_size ==0, \
            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
            dataO = dataO.astype('float32')/255.0
            dataN = dataN.astype('float32')/255.0
            data0 = data0.astype('float32')/255.0
            data1 = data1.astype('float32')/255.0
            indices = list(range(dataO.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_O = dataO[indices[i:i+batch_size]]
                batch_N = dataN[indices[i:i+batch_size]]
                batch_0 = data0[indices[i:i+batch_size]]
                batch_1 = data1[indices[i:i+batch_size]]

                x = [batch_0, batch_1, batch_N]
                y = batch_O
                yield  x,y

# define loss
def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return K.sum(K.square(y_pred - y_true))/2

if __name__ == '__main__':
    # model selection
    import pudb;pu.db
    model = difCNN(depth=15,filters=64)
    model.summary()

    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'model_new_%03d.hdf5'%initial_epoch), compile=False)

    # compile the model
    model.compile(optimizer=Adam(0.001), loss=sum_squared_error)

    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_new_{epoch:03d}.hdf5'),
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)

    #history = model.fit_generator(train_datagen(batch_size=args.batch_size),
    #            steps_per_epoch=2000, epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
    #            callbacks=[checkpointer,csv_logger,lr_scheduler])

    history = model.fit_generator(train_datagen(batch_size=args.batch_size),
                steps_per_epoch=200, epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])

















