import os
import pickle
import numpy as np


#directory = r'J:\_CoverCrop\FUSION\_band_extension\Mosaics\Eight_bands_weighted\Projected\Split\Stacked_uav3Sen8_uav8\5_1_meter\Input_target\Train'
directory = "/home/arifm/usda/PanGAN/data/Data_PAN-GAN/UAV_inputs_NEW"

max_val = []
min_val = []
n = 1
for filename in os.listdir(directory):
    #print(filename)

    path = os.path.join(directory, filename)

    infile = open(path,'rb')
    inPkl = pickle.load(infile)
    infile.close()

    print(inPkl.shape)

    #print(inPkl[1].max(), inPkl[1].min())
    max_val.append(inPkl.max())
    min_val.append(inPkl.min())

    n+=1
print(n)


#print(max_val, min_val)

print(max(max_val), min(max_val))
print(max(min_val), min(min_val))

    

    

    



    
