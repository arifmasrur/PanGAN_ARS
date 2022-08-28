import tensorflow as tf
import numpy as np
import ops as Op

class PanGan(object):
    
    def __init__(self, batch_size,num_spectrum, ratio,init_lr=0.001,lr_decay_rate=0.99,lr_decay_step=1000, is_training=False):
        
        self.num_spectrum=num_spectrum
        self.is_training=is_training
        self.ratio = ratio
        self.batch_size=batch_size
        #self.pan_size=pan_size
        #self.ms_size=ms_size
        self.init_lr=init_lr
        self.lr_decay_rate=lr_decay_rate
        self.lr_decay_step=lr_decay_step
        self.build_model(batch_size,num_spectrum, is_training)
        
    def build_model(self, batch_size, num_spectrum, is_training):
        
        if is_training:
            print("Nothing")

        else:
            with tf.name_scope('input'):
                # self.pan_img=tf.placeholder(dtype=tf.float32, shape=(batch_size, pan_size, pan_size, 1), name='pan_placeholder')
                # self.ms_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,ms_size, ms_size, num_spectrum), name='ms_placeholder')
                
                print("*********************** Here ***********************")
                self.pan_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,None, None, 1), name='pan_placeholder')
                self.ms_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,None, None, num_spectrum), name='ms_placeholder')
                
            
            self.PanSharpening_img=self.PanSharpening_model_dense(self.pan_img, self.ms_img)
            #print(self.PanSharpening_img)
            #self.PanSharpening_img=tf.Print(self.PanSharpening_img, [self.PanSharpening_img])
            
            PanSharpening_img_hp=self.high_pass_1(self.PanSharpening_img)
            pan_img_hp=self.high_pass_1(self.pan_img, 'pan')
            self.pan_img_hp = PanSharpening_img_hp
            
            #PanSharpening_img_hp_print=tf.Print(PanSharpening_img_hp, [PanSharpening_img_hp])
            #print(PanSharpening_img_hp_print)
            
            #pan_img_hp_print=tf.Print(pan_img_hp, [pan_img_hp])
            #print(pan_img_hp_print)
            
            
            ms_img=tf.image.resize_images(self.ms_img, [179, 141], method=1)
            self.ms_img_int = tf.image.resize_images(self.ms_img, [179, 141], method=1)
            
            self.g_spectrum_loss=tf.reduce_mean(tf.square(self.PanSharpening_img-ms_img))
            #self.g_spatial_loss=tf.reduce_mean(tf.square(PanSharpening_img_hp-self.pan_img))
            self.g_spatial_loss=tf.reduce_mean(tf.square(PanSharpening_img_hp-pan_img_hp))
            
            self.g_spectrum_loss = tf.Print (self.g_spectrum_loss, [self.g_spectrum_loss], "--------------------- Spectrum LOSS: ")
            self.g_spatial_loss = tf.Print (self.g_spatial_loss, [self.g_spatial_loss], "--------------------- Spatial LOSS: ")

    
    def PanSharpening_model_dense(self,pan_img, ms_img):
        with tf.variable_scope('Pan_model'):
            if self.is_training:
                with tf.name_scope('upscale'):
                    # de_weight=tf.get_variable('de_weight', [3,3,self.num_spectrum, self.num_spectrum],
                                            # initializer=tf.truncated_normal_initializer(stddev=1e-3) )
                    # ms_scale4 = tf.nn.conv2d_transpose(ms_img, de_weight, output_shape=[self.batch_size,self.pan_size,self.pan_size,self.num_spectrum],
                                                       # strides=[1,4,4,1],padding='SAME' )                            
                    ms_img=tf.image.resize_images(ms_img, [179, 141], method=2)
            
            #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^:", ms_img)
            #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^:", pan_img) 
            
            #pan_img=tf.image.resize_images(pan_img, [5, 5], method=2)
            ms_img=tf.image.resize_images(ms_img, [179, 141], method=1)
            
            input=tf.concat([ms_img,pan_img],axis=-1)
            #input=tf.Print(input, [input])
            #print(input)
            
            
            print("=============================================================")
                    
            with tf.variable_scope('layer1'):
                weights = tf.get_variable("w1", [3, 3, self.num_spectrum + 1, 64],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=1))
                bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
                #conv1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                #                                     decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
                                                     
                                                     
                conv1 = (tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias)
                                                     
                                                     
                #conv1 = tf.Print (conv1, [conv1], "********************* Conv1: ")
                conv1= tf.nn.leaky_relu(conv1)
                
                #conv1 = tf.Print (conv1, [conv1], "--------------------- Conv1: ")
            with tf.variable_scope('layer2'):
                weights = tf.get_variable("w2", [3, 3, 64+9, 32],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=1))
                bias = tf.get_variable("b2", [32], initializer=tf.constant_initializer(0.0))
                #conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(tf.concat([input,conv1],-1), weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                #                                     decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
                                                     
                conv2 = (tf.nn.conv2d(tf.concat([input,conv1],-1), weights, strides=[1, 1, 1, 1], padding='SAME') + bias)
                                                     
                                                     
                conv2= tf.nn.leaky_relu(conv2)
                
            with tf.variable_scope('layer3'):
                weights = tf.get_variable("w3", [3, 3, 9+64+32 , 16],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=1))
                bias = tf.get_variable("b3", [16], initializer=tf.constant_initializer(0.0))
                conv3 = (tf.nn.conv2d(tf.concat([input,conv1,conv2],-1), weights, strides=[1, 1, 1, 1], padding='SAME') + bias)
                conv3= tf.nn.leaky_relu(conv3)
            
        
            #print("^^^^^^^^^^^^^^^^^^^^^^ conv3 shape: ", conv3)   
            
            with tf.variable_scope('layer4'):
                weights = tf.get_variable("w3", [3, 3, 9+64+32+16 , self.num_spectrum],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=1))
                bias = tf.get_variable("b4", [self.num_spectrum], initializer=tf.constant_initializer(0.0))
                conv4 = (tf.nn.conv2d(tf.concat([input,conv1,conv2,conv3],-1), weights, strides=[1, 1, 1, 1], padding='SAME') + bias)
                conv4= tf.nn.relu(conv4)

        return conv4
        
        
   
    def high_pass_1(self, img, type='PanSharepening'):
        if type=='pan':
        
            img = tf.image.resize_images(img, [179, 141], method=2)
            input=img
            for i in range(7):
                input=tf.concat([input,img],axis=-1)
            img=input
        blur_kerel=np.zeros(shape=(3,3,8,8), dtype=np.float32)
        value=np.array([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]])
        for i in range(7):
            blur_kerel[:,:,i,i]=value
        img_hp=tf.nn.conv2d(img, tf.convert_to_tensor(blur_kerel), strides=[1,1,1,1], padding='SAME')
        #img_hp=tf.reshape(tf.reduce_mean(img_hp, 4),[self.batch_size, 35,37, 1])
        #img_hp=img-img_lp
        return tf.abs(img_hp)
        
        

