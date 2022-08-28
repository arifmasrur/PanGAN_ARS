import numpy as np
import os
import h5py
#import gdal
import scipy.io as scio

import PIL
from PIL import Image
import rasterio


    
def read_img1(path):
    img = rasterio.open(path)
    img = img.read()

    img = np.transpose(img, (1, 2, 0))
    img = np.maximum(np.zeros(img.shape), img)
    #h,w=img.shape
    print(img.shape)
    #img = img/np.max(img)
    #img=scio.loadmat(path)['I']
    #img=(img-127.5)/127.5
    img = img.round(4)
    print(np.max(img[:,:,0]))
    print(np.min(img[:,:,0]))
    print(type(img))
    print("******************************** MS read working ********************************")
    return img
    
def read_img2(path):
    print("Here")
    img = Image.open(path)
    print("Here here")
    img = np.array(img)
    img = np.maximum(np.zeros(img.shape), img)
    
    #img = img/np.max(img)
    #img = img.round(4)
    print(np.max(img))
    print(np.min(img))

    print("******************************** Pan read working ********************************")
    return img
    
  
source_path = '/home/arifm/usda/PanGAN/data/test_gt_Image/'
source_ms_path=os.path.join(source_path, 'lrms','BB_Mar19.tif')

read_img1(source_ms_path)
    
#read_img2('/home/arifm/PanGAN/data/Data_PAN-GAN/Pan/1.tif')
    
 
#read_img2('/home/arifm/PanGAN/data/test_gt_NEW/pan/1.TIF')                
