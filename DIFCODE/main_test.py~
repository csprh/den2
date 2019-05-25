# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/husqin/DnCNN-keras
# =============================================================================

# run this to test the model

import argparse
import os, time, datetime
#import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from keras.preprocessing import image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='../data/AWGN25/Test', type=str, help='directory of test dataset')
    parser.add_argument('--orig_dir', default='../data/Origs/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set68','Set12'], type=list, help='name of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models','difCNN_sigma25'), type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_new_037.hdf5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()

def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def save_result(result,path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt','.dlm'):
        np.savetxt(path,result,fmt='%2.4f')
    else:
        imsave(path,np.clip(result,0,1))


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':

    import pudb;pu.db
    args = parse_args()

    model = load_model(os.path.join(args.model_dir, args.model_name),compile=False)
    log('load trained model')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir,set_cur)):
            os.mkdir(os.path.join(args.result_dir,set_cur))
        psnrs = []
        ssims = []

        for im in os.listdir(os.path.join(args.orig_dir,set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                #x = np.array(Image.open(os.path.join(args.set_dir,set_cur,im)), dtype='float32') / 255.0
                #x = np.array(imread(os.path.join(args.orig_dir,set_cur,im)), dtype=np.float32) / 255.0
                #y = np.array(imread(os.path.join(args.set_dir,set_cur,'NoiseOrigs',im)), dtype=np.float32) / 255.0
                #d0 = np.array(imread(os.path.join(args.orig_dir,set_cur,'Denoise0',im)), dtype=np.float32) / 255.0
                #d1 = np.array(imread(os.path.join(args.set_dir,set_cur,'Denoise1',im)), dtype=np.float32) / 255.0

        #        data1 = dataO.astype('float32')/255.0
                x = image.load_img(os.path.join(args.orig_dir,set_cur,im),grayscale=True); x = image.img_to_array(x)/255.0
                y = image.load_img(os.path.join(args.set_dir,set_cur,'NoiseOrigs',im),grayscale=True); y = image.img_to_array(y)/255.0
                d0 = image.load_img(os.path.join(args.set_dir,set_cur,'Denoise0',im),grayscale=True); d0 = image.img_to_array(d0)/255.0
                d1 = image.load_img(os.path.join(args.set_dir,set_cur,'Denoise1',im),grayscale=True); d1 = image.img_to_array(d1)/255.0
                yt  = to_tensor(y)
                d0t  = to_tensor(d0)
                d1t  = to_tensor(d1)
                start_time = time.time()
                x_ = model.predict([d0t,d1t]) # inference
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second'%(set_cur,im,elapsed_time))
                x_=from_tensor(x_)
                x_=np.clip(x_,0,1)
                psnr_x_ = compare_psnr(np.squeeze(x), x_)
                save_result(x_,'test1.png')

                psnr_x_ = compare_psnr(np.squeeze(x), np.squeeze(x_))
                psnr_d0 = compare_psnr(np.squeeze(x), np.squeeze(d0))
                psnr_d1 = compare_psnr(np.squeeze(x), np.squeeze(d1))
                ssim_x_ = compare_ssim(np.squeeze(x), np.squeeze(x_))
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    show(np.hstack((y,x_))) # show the image
                    save_result(x_,path=os.path.join(args.result_dir,set_cur,name+'_dncnn'+ext)) # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        if args.save_result:
            save_result(np.hstack((psnrs,ssims)),path=os.path.join(args.result_dir,set_cur,'results.txt'))

        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))




