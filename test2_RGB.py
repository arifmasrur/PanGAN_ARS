import tensorflow as tf
import os 
import numpy as np
#import gdal
#import cv2
#from PanGan_2 import PanGan
from PanGan_test_RGB import PanGan
from DataSet_RGB import DataSet
from config_RGB import FLAGES
import scipy.io as scio
import time
import os
import tifffile
import pickle

import PIL
from PIL import Image
import rasterio

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''定义参数'''

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('pan_size',
                           default_value=None,
                           docstring='pan image size')
tf.app.flags.DEFINE_string('ms_size',
                           default_value=None,
                           docstring='ms image size')
tf.app.flags.DEFINE_integer('batch_size',
                           default_value=1,
                           docstring='img batch')
tf.app.flags.DEFINE_integer('num_spectrum',
                           default_value=3,
                           docstring='spectrum num')
tf.app.flags.DEFINE_integer('ratio',
                           default_value=5,
                           docstring='pan image/ms img')
tf.app.flags.DEFINE_string('model_path',
                           default_value='/home/arifm/new_usda/PanGan_RGB/data/Data_PAN-GAN/model_11_25-generator/Generator-9500',

                           docstring='pan image/ms img') 
tf.app.flags.DEFINE_string('test_path',
                           default_value='./data/test_gt_Image',
                           docstring='test img data')                            
tf.app.flags.DEFINE_string('result_path',
                           default_value='./result',
                           docstring='result img')                          
tf.app.flags.DEFINE_bool('norm',
                           default_value=True,
                           docstring='if norm') 


                           
def main(argv):
    if not os.path.exists(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)
    model=PanGan(FLAGS.batch_size, FLAGS.num_spectrum, FLAGS.ratio,0.001, 0.99, 1000,False)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.model_path)
        ms_test_path= FLAGS.test_path + '/lrms'
        pan_test_path=FLAGS.test_path + '/pan'
        hrms_gt_path =FLAGS.test_path + '/hrms_gt'
        for img_name in os.listdir(ms_test_path):
            start=time.time()
            print(img_name)
            pan, ms, hrms_img = read_img(pan_test_path, ms_test_path, hrms_gt_path, img_name, FLAGS)
            print("----------------------------------- ", pan.shape)
            
            start=time.time()
            PanSharpening,ms_img_int, pan_hp, error,error2= sess.run([model.PanSharpening_img, model.ms_img_int, model.pan_img_hp, model.g_spectrum_loss, model.g_spatial_loss], feed_dict={model.pan_img:pan, model.ms_img:ms})
            #PanSharpening=PanSharpening*127.5+127.5
            
            print("+++++++++++++++++++++ ", type(PanSharpening), PanSharpening.shape, ms.shape)
            #PanSharpening=PanSharpening*np.max(PanSharpening)
            #PanSharpening=PanSharpening.squeeze()
            #PanSharpening=PanSharpening.astype('uint8')
            end=time.time()
            print(end-start)
            save_name=img_name.split('.')[0] + '.tif'
            save_path=os.path.join(FLAGS.result_path, save_name)
            #cv2.imwrite(save_path, PanSharpening)
            #img_write(PanSharpening,save_path)
            # PanSharpening=cv2.cvtColor(PanSharpening[:,:,0:3], cv2.COLOR_BGR2RGB)
            # cv2.imwrite(save_path, PanSharpening)
            
            
            with rasterio.open('/home/arifm/new_usda/PanGan_RGB/data/test_gt_Image/hrms_gt/MV_Apr19.tif') as src:
              ras_meta = src.profile
              print(ras_meta)
              
            print(PanSharpening.shape)
            PanSharpening1 = PanSharpening[0]
            h,w,c=PanSharpening1.shape
            lrms_sharpened=np.transpose(PanSharpening1, (2, 0, 1))
              
            with rasterio.open('/home/arifm/new_usda/PanGan_RGB/result/MV_Apr19_sharpened_RGB.tif', 'w', **ras_meta) as dst:
              dst.write(lrms_sharpened)
              
            
            #test_out = open(os.path.join('/home/arifm/usda/PanGAN/result', "MV_Apr19_sharpened.pkl") ,'wb')
            #pickle.dump(PanSharpening1, test_out)
            
            
            _, axarr = plt.subplots(2, PanSharpening.shape[3],
                                    figsize=(PanSharpening.shape[3] * 5, 8))
            for t in range(PanSharpening.shape[3]):
                plt.title("LRMS 5-m")
                axarr[0][t].imshow(ms[0, :, :, t], cmap='gray')
                plt.title("Predicted HRMS 1-m")
                axarr[1][t].imshow(PanSharpening[0, :, :, t], cmap='gray')
                #axarr[3][t].imshow(pan_hp[0, :, :, t], cmap='viridis')
                #plt.title("HRMS 1-m")
                #axarr[2][t].imshow(hrms_img[0, :, :, t], cmap='gray')
                
            plt.savefig(os.path.join(FLAGS.result_path, '{name}_testing.png'.format(name=save_name)))

            plt.close()
            
            #tifffile.imsave(save_path, PanSharpening)
            print(img_name + ' DONE.' + 'Spectrum error is: ' + str(error) + ' Spatial error is: ' + str(error2))
 
 
 

#def read_img(pan_test_path, ms_test_path, img_name, FLAGS):
#    pan_img_path=os.path.join(pan_test_path, img_name)
#    ms_img_path=os.path.join(ms_test_path, img_name)
#
#    #path = os.path.join(directory, filename)
#    infile = open(pan_img_path,'rb')
#    inPkl = pickle.load(infile)
#    infile.close()
#    
#    pan_img=inPkl
#    #pan_img = np.mean(pan_img, axis=2)
#    h,w=pan_img.shape
#    pan_img=pan_img.reshape((1,h,w,1))
#
#    #path = os.path.join(directory, filename)
#    infile1 = open(ms_img_path,'rb')
#    inPkl1 = pickle.load(infile1)
#    infile1.close()
#    
#    ms_img=inPkl1
#    h,w,c=ms_img.shape
#    ms_img=ms_img.reshape((1,h,w,c))
#    return pan_img, ms_img


          
def read_img(pan_test_path, ms_test_path, hrms_gt_path, img_name, FLAGS):
    pan_img_path=os.path.join(pan_test_path, img_name)
    ms_img_path=os.path.join(ms_test_path, img_name)
    hrms_img_path=os.path.join(hrms_gt_path, img_name)

    pan_img=read_img1(pan_img_path)
    #pan_img = np.mean(pan_img, axis=2)
    h,w,c=pan_img.shape
    pan_img=pan_img.reshape((1,h,w,c))

    ms_img=read_img1(ms_img_path)
    h,w,c=ms_img.shape
    ms_img=ms_img.reshape((1,h,w,c))
    
    hrms_img=read_img1(hrms_img_path)
    h,w,c=hrms_img.shape
    hrms_img=hrms_img.reshape((1,h,w,c))
    
    
    return pan_img, ms_img, hrms_img
    
#def gdal_read(path,name):
#    data=gdal.Open(path)
#    w=data.RasterXSize
#    h=data.RasterYSize
#    img=data.ReadAsArray(0,0,w,h)
#    if name == 'ms':
#        img=np.transpose(img,(1,2,0))
#    img=(img-1023.5)/1023.5
#    return img
#    
#def read8bit(path,name):
#    if name=='ms':
#        v='src'
#    else:
#        v='pan'
#    v='I'
#    #img=scio.loadmat(path)[v]
#    img=np.load(path)
#    img=(img-127.5)/127.5
#    return img
    

def read_img1(path):
  img = rasterio.open(path)
  img = img.read()

  img = np.transpose(img, (1, 2, 0))
  img = np.maximum(np.zeros(img.shape), img)
  #img = img/np.max(img)
  #img=scio.loadmat(path)['I']
  #img=(img-127.5)/127.5
  #img = img.round(8)
  print(np.max(img[:,:,0]))
  print(np.min(img[:,:,0]))
  #print(type(img))
  print("******************************** MS read working ********************************")
  return img
  
  
def read_img2(path):
  img = rasterio.open(path)
  img = img.read()

  img = np.transpose(img, (1, 2, 0))
  img = np.maximum(np.zeros(img.shape), img)
  #img = img/np.max(img)
  #img=scio.loadmat(path)['I']
  #img=(img-127.5)/127.5
  #img = img.round(8)
  #print(np.max(img[:,:,0]))
  #print(np.min(img[:,:,0]))
  #print(type(img))
  print("******************************** MS read working ********************************")
  return img
  
  
def read_img3(path):
  print("Here")
  img = Image.open(path)
  print("Here here")
  img = np.array(img)
  img = np.maximum(np.zeros(img.shape), img)
  img = img/np.max(img)
  img = img.round(8)
  print(np.max(img))
  print(np.min(img))

  print("******************************** Pan read working ********************************")
  return img
      
      
      
    
#def img_write(img_array,save_path):
#    datatype=gdal.GDT_UInt16
#    h,w,c=img_array.shape
#    driver=gdal.GetDriverByName('GTiff')
#    data=driver.Create(save_path, w, h, c, datatype)
#    for i in range(c):
#        data.GetRasterBand(i+1).WriteArray(img_array[:,:,i])
#    del data
    
    
    
    
if __name__ == '__main__':
    tf.app.run()
    
      
    
