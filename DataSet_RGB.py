import numpy as np
import os
import h5py
#import gdal
import scipy.io as scio

import PIL
from PIL import Image
import rasterio
import pickle
import random


class DataSet(object):
    def __init__(self, pan_size, uav_size, ms_size, source_path, data_save_path, batch_size, stride, category='train'):
        self.pan_size=pan_size
        self.uav_size=uav_size
        self.ms_size=ms_size

        
        self.batch_size=batch_size
        if not os.path.exists(data_save_path):
            self.make_data(source_path,data_save_path,stride)
        self.pan, self.uav, self.ms =self.read_data(data_save_path, category)
        self.data_generator=self.generator()
        data_generator = self.data_generator 
        
        
    def generator(self):
        num_data=self.pan.shape[0]
        print(num_data)
        while True:
            batch_pan=np.zeros((self.batch_size,self.pan_size,self.pan_size,1))
            batch_uav=np.zeros((self.batch_size,self.uav_size,self.uav_size,3))
            batch_ms=np.zeros((self.batch_size,self.ms_size,self.ms_size,3))
           
            
            
            for i in range(self.batch_size):
                random_index=np.random.randint(0,num_data)
                batch_pan[i]=self.pan[random_index]
                batch_uav[i]=self.uav[random_index]
                batch_ms[i]=self.ms[random_index]
               
            yield batch_pan, batch_uav, batch_ms
    
    def read_data(self,path,category):
        f=h5py.File(path, 'r')
        if category == 'train':
            pan=np.array(f['pan_train'])
            uav=np.array(f['uav_train'])
            ms=np.array(f['ms_train'])
            
        else:
            pan=np.array(f['pan_valid'])
            uav=np.array(f['uav_valid'])
            ms=np.array(f['ms_valid'])
            
        return pan, uav, ms
        
        
    
    def get_file_paths(self, root):
      print("The train directory is {}".format(root))
      f = []
      for _,_,filename in os.walk(root):
          f.extend(filename)
          #print(filename)
      f = sorted(f)
      paths = []
      i = 1 
      for file in f:
        #print(file)
        filepath = paths.append(os.path.join(root,file))
        i+=1
    
      
      print("Total paths: " + str(len(paths)))
      return paths
      
      
    def all_data(self, root_ms, root_uav, root_pan):
        paths_lrms = self.get_file_paths(root_ms)
        paths_uav = self.get_file_paths(root_uav)
        paths_pan = self.get_file_paths(root_pan)
        
        all_lrms = []
        all_uav = []
        all_pan  = []
    
        for i, path in enumerate(paths_lrms):
            #print(path)
            pkl_data1 = pickle.load(open(path,"rb"))
            pkl_data = np.where(pkl_data1>100, 0, pkl_data1)
            #print(path, pkl_data.shape)
            all_lrms.append(pkl_data)
            
        for i, path in enumerate(paths_uav):
            #print(path)
            pkl_data1 = pickle.load(open(path,"rb"))
            pkl_data = np.where(pkl_data1>100, 0, pkl_data1)
            #print(path, pkl_data.shape)
            all_uav.append(pkl_data)
    
    
        for i, path in enumerate(paths_pan):
            #print(path)
            pkl_data1 = pickle.load(open(path,"rb"))
            pkl_data2 = np.where(pkl_data1>100, 0, pkl_data1)
            pkl_data = pkl_data2.reshape(pkl_data2.shape[0], pkl_data2.shape[1], 1)
            #print(path, pkl_data.shape)
            all_pan.append(pkl_data)
    
        print(len(all_lrms), len(all_uav), len(all_pan))
    
        allData = [all_lrms, all_uav, all_pan]
        return allData    
    

        
    def make_data(self, source_path, data_save_path, stride):
        # source_ms_path=os.path.join(source_path, 'MS','1.TIF')
        # source_pan_path=os.path.join(source_path, 'Pan','1.TIF')
        #source_ms_path=os.path.join(source_path, 'MS_inputs_NEW')
        source_ms_path=os.path.join(source_path, 'LRMS')
        #print(source_ms_path)
        #source_pan_path=os.path.join(source_path, 'Pan_inputs_NEW')
        source_pan_path=os.path.join(source_path, 'PAN_HRMS')
        #source_uav_path=os.path.join(source_path, 'UAV_inputs_NEW')
        source_uav_path=os.path.join(source_path, 'HRMS')
        

        #all_pan=self.crop_to_patch(source_pan_path, stride, name='pan')
        #all_ms=self.crop_to_patch(source_ms_path, stride, name='ms')
        all_in_data = self.all_data(source_ms_path, source_uav_path, source_pan_path)

        all_ms = all_in_data[0]
        all_uav = all_in_data[1]
        all_pan = all_in_data[2]
        
        
        print('The number of ms patch is: ' + str(len(all_ms)))
        print('The number of UAV patch is: ' + str(len(all_uav)))
        print('The number of pan patch is: ' + str(len(all_pan)))
        
        pan_train, pan_valid, uav_train, uav_valid, ms_train, ms_valid=self.split_data(all_pan, all_uav, all_ms)
        print('The number of pan_train patch is: ' + str(len(pan_train)))
        print('The number of pan_valid patch is: ' + str(len(pan_valid)))

        print('The number of UAV_train patch is: ' + str(len(uav_train)))
        print('The number of UAV_valid patch is: ' + str(len(uav_valid)))
        
        print('The number of ms_train patch is: ' + str(len(ms_train)))
        print('The number of ms_valid patch is: ' + str(len(ms_valid)))
        
        pan_train=np.array(pan_train)
        pan_valid=np.array(pan_valid)
        
        uav_train=np.array(uav_train)
        uav_valid=np.array(uav_valid)
        
        ms_train=np.array(ms_train)
        ms_valid=np.array(ms_valid)
        
        f=h5py.File(data_save_path,'w')
        
        f.create_dataset('pan_train', data=pan_train)
        f.create_dataset('pan_valid', data=pan_valid)
        f.create_dataset('uav_train', data=uav_train)
        f.create_dataset('uav_valid', data=uav_valid)
        f.create_dataset('ms_train', data=ms_train)
        f.create_dataset('ms_valid', data=ms_valid)
        

        
    def split_data(self,all_pan, all_uav, all_ms):
        ''' all_pan和all_ms均为list'''
        pan_train=[]
        pan_valid=[]
        
        uav_train =[]
        uav_valid =[]
        
        ms_train=[]
        ms_valid=[]
        
        for i in range(len(all_pan)):
            #print(i)
            if i < 6438:
                rand=np.random.randint(0,100)
                if rand <=10:
                    pan_valid.append(all_pan[i])
                    uav_valid.append(all_uav[i])
                    ms_valid.append(all_ms[i])
                else:
                    ms_train.append(all_ms[i])
                    uav_train.append(all_uav[i])
                    pan_train.append(all_pan[i])
        return pan_train, pan_valid, uav_train, uav_valid, ms_train, ms_valid
        
           
#    def crop_to_patch(self, img_path, stride, name):
#        #img=(cv2.imread(img_path,-1)-127.5)/127.5
#
#        all_img=[]
#        if name == 'ms':
#            img=self.read_img1(img_path)
#            h=img.shape[0]
#            w=img.shape[1]
#            print(h)
#            print(w)
#            for i in range(0, h-self.ms_size, stride):
#                for j in range(0, w-self.ms_size, stride):
#                    img_patch=img[i:i+self.ms_size, j:j+self.ms_size,:]
#                    all_img.append(img_patch)
#                    if i + self.ms_size >= h:
#                        img_patch=img[h-self.ms_size:, j:j+self.ms_size,:]
#                        all_img.append(img_patch)
#                img_patch=img[i:i+self.ms_size, w-self.ms_size:,:]
#                all_img.append(img_patch)
#        else:
#            print(img_path)
#            img=self.read_img2(img_path)
#            h=img.shape[0]
#            w=img.shape[1]
#            print(h)
#            print(w)
#            for i in range(0, h-self.pan_size, stride*2):
#                for j in range(0, w-self.pan_size, stride*2):
#                    img_patch=img[i:i+self.pan_size, j:j+self.pan_size].reshape(self.pan_size,self.pan_size,1)
#                    all_img.append(img_patch)
#                    if i + self.pan_size >= h:
#                        img_patch=img[h-self.pan_size:, j:j+self.pan_size].reshape(self.pan_size,self.pan_size,1)
#                        all_img.append(img_patch)
#                img_patch=img[i:i+self.pan_size, w-self.pan_size:].reshape(self.pan_size,self.pan_size,1)
#                all_img.append(img_patch)
#        return all_img      
        
#    def read_img(self,path,name):
#        data=gdal.Open(path)
#        w=data.RasterXSize
#        h=data.RasterYSize
#        img=data.ReadAsArray(0,0,w,h)
#        if name == 'ms':
#            img=np.transpose(img,(1,2,0))
#        img=(img-1023.5)/1023.5
#        return img
#        
#    def read_img1(self, path):
#      img = rasterio.open(path)
#      img = img.read()
#    
#      img = np.transpose(img, (1, 2, 0))
#      img = np.maximum(np.zeros(img.shape), img)
#      img = img/np.max(img)
#      #img=scio.loadmat(path)['I']
#      #img=(img-127.5)/127.5
#      img = img.round(4)
#      print(np.max(img[:,:,0]))
#      print(np.min(img[:,:,0]))
#      #print(type(img))
#      print("******************************** MS read working ********************************")
#      return img
#      
#    def read_img2(self, path):
#      print("Here")
#      img = Image.open(path)
#      print("Here here")
#      img = np.array(img)
#      img = np.maximum(np.zeros(img.shape), img)
#      img = img/np.max(img)
#      img = img.round(4)
#      print(np.max(img))
#      print(np.min(img))
#    
#      print("******************************** Pan read working ********************************")
#      return img
    

        
                    
         
    
    
 
                
