class FLAGES(object):

    pan_size= 25
    
    ms_size=5
    
    uav_size = 25
    
    
    num_spectrum=8
    
    ratio=5
    stride=1
    norm=True
    
    
    batch_size=1
    lr=0.0001
    decay_rate=0.99
    decay_step=10000
    
    img_path='/home/arifm/new_usda/data/Data_PAN-GAN'
    data_path='/home/arifm/new_usda/data/Data_PAN-GAN/train_qk.h5'
    log_dir='/home/arifm/new_usda/data/Data_PAN-GAN/log_11_25-generator'
    model_save_dir='/home/arifm/new_usda/data/Data_PAN-GAN/model_11_25-generator'
    
    is_pretrained=False
    
    #iters=500000
    iters=100000
    model_save_iters = 500
    valid_iters=10