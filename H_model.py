from pickle import TRUE
from tf_spatial_transform import transform
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import math
#from tensorflow.compat.v1.layers import Conv2D,Dense,Flatten,Dropout
import numpy as np
import tf_slim as slim
from tensorDLT import solve_DLT
#from tensorflow.keras import layers
#from tensorflow import keras
from swin_trainsformer import SwinTransformer
from swin_trainsformer import PatchExtract, PatchEmbedding, PatchMerging
#from tensorflow.keras.layers import conv2d
def CCL(c1, warp):
    shape = warp.get_shape().as_list()
    kernel = 3
    stride = 1
    rate = 1
 
    # extract patches as convolutional filters
    patches = tf.compat.v1.extract_image_patches(warp, [1,kernel,kernel,1], [1,stride,stride,1], [1,rate,rate,1], padding='SAME')
    patches = tf.reshape(patches, [shape[0], -1, kernel, kernel, shape[3]])
    matching_filters = tf.transpose(patches, [0, 2, 3, 4, 1])
    print(matching_filters.shape)
    
    # using convolution to match
    match_vol = []
    for i in range(shape[0]):
      single_match = tf.nn.atrous_conv2d(tf.expand_dims(c1[i], [0]), matching_filters[i], rate=rate, padding='SAME')
      match_vol.append(single_match)
    
    match_vol = tf.concat(match_vol, axis=0)
    channels = int(match_vol.shape[3])

    print("channels")
    print(channels)
    
    # scale softmax
    softmax_scale = 10
    match_vol = tf.nn.softmax(match_vol*softmax_scale,3)
    
    # convert the correlation volume to feature flow
    # h_one = tf.linspace(0., shape[1]-1., int(match_vol.shape[1]))
    # w_one = tf.linspace(0., shape[2]-1., int(match_vol.shape[2]))
    # h_one = tf.matmul(tf.expand_dims(h_one, 1), tf.ones(shape=tf.stack([1, shape[2]])))
    # w_one = tf.matmul(tf.ones(shape=tf.stack([shape[1], 1])), tf.transpose(tf.expand_dims(w_one, 1), [1, 0]))
    # h_one = tf.tile(tf.expand_dims(tf.expand_dims(h_one, 0),3), [shape[0],1,1,channels])
    # w_one = tf.tile(tf.expand_dims(tf.expand_dims(w_one, 0),3), [shape[0],1,1,channels])
    
    # i_one = tf.expand_dims(tf.linspace(0., channels-1., channels),0)
    # i_one = tf.expand_dims(i_one,0)
    # i_one = tf.expand_dims(i_one,0)
    # i_one = tf.tile(i_one, [shape[0], shape[1], shape[2], 1])
 
    # flow_w = match_vol*(i_one%shape[2] - w_one)
    # flow_h = match_vol*(i_one//shape[2] - h_one)
    # flow_w = tf.expand_dims(tf.reduce_sum(flow_w,3),3)
    # flow_h = tf.expand_dims(tf.reduce_sum(flow_h,3),3)
    
    # flow = tf.concat([flow_w, flow_h], 3)
    # print("flow.shape")
    # print(flow.shape)

    return match_vol
def SE_block(x_0, r):
    channels = x_0.shape[-1]
    batch_size, height, width, channels = x_0.get_shape().as_list()
    assert height==width
    with tf.compat.v1.variable_scope('se_block'):
    #x = tf.compat.v1.layers.MaxPooling2D(pool_size=height,strides=0)(x_0)
      x = tf.nn.max_pool2d(x_0,ksize=[1,x_0.get_shape().as_list()[1],x_0.get_shape().as_list()[1],1],strides=[1,1,1,1],padding='VALID')
   
      #x = x[:, None, None, :]

     
      x=tf.compat.v1.layers.Conv2D(filters=channels//r, kernel_size=1, strides=1,activation=tf.nn.relu)(x)
    # #x = Activation('relu')(x)
      x = tf.compat.v1.layers.Conv2D(filters=channels, kernel_size=1, strides=1,activation=tf.nn.sigmoid)(x)
      #x = tf.nn.sigmoid(x)
      x = x_0*x
    
    return x

def crossNonLocalBlock(input_x,input_y,output_channels,lamda, scope="NonLocalBlock"):
     batch_size, height, width, in_channels = input_x.get_shape().as_list()
     with tf.compat.v1.variable_scope(scope):
        with tf.compat.v1.variable_scope("g",reuse=None):
            g1 = tf.compat.v1.layers.Conv2D(filters=output_channels, kernel_size=1, strides=(1,1), padding='same')(input_x)
            #print(g1.shape)
        with tf.compat.v1.variable_scope("g",reuse=True):
            g2 = tf.compat.v1.layers.Conv2D(filters=output_channels, kernel_size=1, strides=(1,1), padding='same')(input_y)
        with tf.compat.v1.variable_scope("phi",reuse=None):
            phi1 =tf.compat.v1.layers.Conv2D(filters=output_channels, kernel_size=1, strides=(1,1), padding='same')(input_x)
        with tf.compat.v1.variable_scope("phi",reuse=True):
            phi2 = tf.compat.v1.layers.Conv2D(filters=output_channels, kernel_size=1, strides=(1,1), padding='same')(input_y) 
        with tf.compat.v1.variable_scope("theta",reuse=None):
            theta1 = tf.compat.v1.layers.Conv2D(filters=output_channels, kernel_size=1, strides=(1,1), padding='same')(input_x)
        with tf.compat.v1.variable_scope("theta",reuse=True):
            theta2 = tf.compat.v1.layers.Conv2D(filters=output_channels, kernel_size=1, strides=(1,1), padding='same')(input_y)
         

        #g_x = tf.reshape(g, [-1, output_channels, height * width])
        g_x2 = tf.reshape(g2, [-1, height * width, output_channels])
        g_x1 = tf.reshape(g1, [-1, height * width, output_channels])
        #g_x = tf.transpose(g_x, [0, 2, 1])
        #print(g_x.shape)

        phi_x2 = tf.reshape(phi2, [-1, output_channels, height * width])
        phi_x1 = tf.reshape(phi1, [-1, output_channels, height * width])
        #print(phi_x.shape)
        phi_x2 = tf.nn.l2_normalize(phi_x2, axis = -1)
        phi_x1 = tf.nn.l2_normalize(phi_x1, axis = -1)
        #theta_x = tf.reshape(theta, [-1, output_channels, height * width])
        #theta_x = tf.transpose(theta_x, [0, 2, 1])
        theta_x1 = tf.reshape(theta1, [-1, height * width, output_channels])
        theta_x2 = tf.reshape(theta2, [-1, height * width, output_channels])
        #print(theta_x1.shape)
        theta_x1 = tf.nn.l2_normalize(theta_x1, axis = -2)
        theta_x2 = tf.nn.l2_normalize(theta_x2, axis = -2)

        f1 = tf.matmul(theta_x1, phi_x2)
        f1_softmax = tf.nn.softmax(f1*10, -1)   
        y1 = tf.matmul(f1_softmax, g_x2)
        f2 = tf.matmul(theta_x2, phi_x1)
        f2_softmax = tf.nn.softmax(f2*10, -1)      
        y2 = tf.matmul(f2_softmax, g_x1)
        y1 = tf.reshape(y1, [-1, height, width, output_channels])
        y2 = tf.reshape(y2, [-1, height, width, output_channels])

        with tf.compat.v1.variable_scope("w",reuse=None):
            w_y1 = tf.compat.v1.layers.Conv2D(filters=in_channels, kernel_size=1, strides=(1,1), padding='same')(y1)
        with tf.compat.v1.variable_scope("w",reuse=True):
            w_y2 = tf.compat.v1.layers.Conv2D(filters=in_channels, kernel_size=1, strides=(1,1), padding='same')(y2)    
        z1 = lamda*input_x + (1-lamda)*w_y1
        z2 = lamda*input_y + (1-lamda)*w_y2

        return z1,z2

def H_model(inputs_aug, inputs, is_training, patch_size=128.):

    batch_size = tf.shape(input=inputs)[0]
    net1_f, net2_f, net3_f,feature11,feature21   = build_model(inputs_aug, is_training)
    
    
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    
    H1 = solve_DLT(net1_f, patch_size)
    H2 = solve_DLT(net1_f+net2_f, patch_size)
    H3 = solve_DLT(net1_f+net2_f+net3_f, patch_size)
    
    H1_mat = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    H2_mat = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    H3_mat = tf.matmul(tf.matmul(M_tile_inv, H3), M_tile)
    
    image2_tensor = inputs[..., 3:6]
    warp2_H1 = transform(image2_tensor, H1_mat)
    warp2_H2 = transform(image2_tensor, H2_mat)
    warp2_H3 = transform(image2_tensor, H3_mat)
    feature2_H1 = transform(feature21, H1_mat)
     
    
    one = tf.ones_like(image2_tensor, dtype=tf.float32)
    one_warp_H1 = transform(one, H1_mat)
    one_warp_H2 = transform(one, H2_mat)
    one_warp_H3 = transform(one, H3_mat)
    one_1 = tf.ones_like(feature11, dtype=tf.float32)
    one_1_warp_H1 = transform(one_1, H1_mat)
    #one_2 = tf.ones_like(feature12, dtype=tf.float32)
    #one_2_warp_H2 = transform(one_2, H2_mat)
    
    return net1_f, net2_f, net3_f, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3,one_1_warp_H1,feature11,feature2_H1

def H_model_v2(inputs, is_training):
    net1_f, net2_f, net3_f = build_model(inputs, is_training)  
    shift = net1_f + net2_f + net3_f 
    
    return shift


def build_model(inputs, is_training):
    with tf.compat.v1.variable_scope('model'):
        input1 = inputs[...,0:3]
        input2 = inputs[...,3:6]
        #resize to 128*128
        input1 = tf.image.resize(input1, [128,128],method='bilinear')
        input2 = tf.image.resize(input2, [128,128],method='bilinear')
        input1 = tf.expand_dims(tf.reduce_mean(input_tensor=input1, axis=3),[3])
        input2 = tf.expand_dims(tf.reduce_mean(input_tensor=input2, axis=3),[3])
        net1_f, net2_f, net3_f,feature11,feature21 = _H_model(input1, input2, is_training)
        return net1_f, net2_f, net3_f,feature11,feature21 

#def _conv_block(x, num_out_layers, kernel_sizes, strides):
 #   conv1 = tf.keras.layers.Conv2D(filters=num_out_layers[0], kernel_size=kernel_sizes[0], activation=tf.nn.relu)(x)
 #   conv2 = tf.keras.layers.Conv2D(filters=num_out_layers[1], kernel_size=kernel_sizes[1], activation=tf.nn.relu)(conv1)
 #   return conv2

def feature_extractor(image_tf,is_training):
    feature = []
    prob = 0.03 if is_training==True else 0.0
    #image_tf = tf.expand_dims(image_tf, [3])
    with tf.compat.v1.variable_scope('conv_block1'): # H
      conv1_1 = tf.compat.v1.layers.Conv2D(filters=64, kernel_size=3, padding='same',activation=tf.nn.relu)(image_tf)
      conv1_2 = tf.compat.v1.layers.Conv2D(filters=64, kernel_size=3, padding='same',activation=tf.nn.relu)(conv1_1) 
      #conv1 = _conv_block(image_tf, ([64, 64]), (3, 3), (1, 1))
      feature.append(conv1_2)
      #maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding = 'SAME')(conv1_2)
    with tf.compat.v1.variable_scope('conv_block2'):
      #conv2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu)(maxpool1)
      #conv2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu)(conv2_1)
      #conv2 = _conv_block(maxpool1, ([64, 64]), (3, 3), (1, 1))
      # maxpool_patch = PatchExtract((2,2))(conv1_2)
      # conv2_emdedding = PatchEmbedding(64 * 64, 96)(maxpool_patch)
      conv2_embedding = PatchEmbedding()(conv1_2)
      conv2_1 = SwinTransformer(
      dim=96,
      num_patch=(64, 64),
      num_heads=3,
      window_size=4,
      shift_size=0,
      num_mlp=96*4,
      qkv_bias=TRUE,
      dropout_rate=prob,
       )(conv2_embedding)
      conv2_2 = SwinTransformer(
      dim=96,
      num_patch=(64, 64),
      num_heads=3,
      window_size=4,
      shift_size=2,
      num_mlp=96*4,
      qkv_bias=TRUE,
      dropout_rate=prob,
      )(conv2_1)
      # maxpool1= tf.compat.v1.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(conv1_2)
      # conv2_1 = tf.compat.v1.layers.Conv2D(filters=96, kernel_size=3, padding='same',activation=tf.nn.relu)(maxpool1)
      # conv2_2 = tf.compat.v1.layers.Conv2D(filters=96, kernel_size=3, padding='same',activation=tf.nn.relu)(conv2_1)
      conv1_r64 = tf.image.resize(conv1_2, [64, 64])
      conv2_3=tf.reshape(conv2_2,[1,64,64,-1])
      conv2 = tf.compat.v1.layers.Conv2D(filters=96, kernel_size=3, padding='same',activation=tf.nn.relu)(conv2_3)
      feature.append(tf.concat([conv2, conv1_r64], 3))
      maxpool2 = PatchMerging((64, 64), embed_dim=96)(conv2_2)
      #maxpool2 =tf.compat.v1.layers.MaxPooling2D(pool_size=2, strides=2, padding = 'same')(conv2)
    with tf.compat.v1.variable_scope('conv_block3'):
      #conv3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)(maxpool2)
      #conv3_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)(conv3_1)
      #conv3 = _conv_block(maxpool2, ([128, 128]), (3, 3), (1, 1))
      # conv3_emdedding = PatchEmbedding(32* 32, 96*2)(maxpool2)
      conv3_1 = SwinTransformer(
      dim=96*2,
      num_patch=(32, 32),
      num_heads=6,
      window_size=4,
      shift_size=0,
      num_mlp=96*4*2,
      qkv_bias=TRUE,
      dropout_rate=prob,
       )(maxpool2)
      conv3_2 = SwinTransformer(
      dim=96*2,
      num_patch=(32, 32),
      num_heads=6,
      window_size=4,
      shift_size=2,
      num_mlp=96*4*2,
      qkv_bias=TRUE,
      dropout_rate=prob,
      )(conv3_1)
      # conv3_1 = tf.compat.v1.layers.Conv2D(filters=96*2, kernel_size=3, padding='same',activation=tf.nn.relu)(maxpool2)
      # conv3_2 = tf.compat.v1.layers.Conv2D(filters=96*2, kernel_size=3, padding='same',activation=tf.nn.relu)(conv3_1)
      conv1_r32 = tf.image.resize(conv1_2, [32, 32])
      conv3_3=tf.reshape(conv3_2,[1,32,32,-1])
      #conv3_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',activation=tf.nn.relu)(conv3_2)
      conv3 = tf.compat.v1.layers.Conv2D(filters=96*2, kernel_size=3, padding='same',activation=tf.nn.relu)(conv3_3)
      feature.append(tf.concat([conv3, conv1_r32], 3))
      maxpool3 = PatchMerging((32, 32), embed_dim=96*2)(conv3_2)
      #
    with tf.compat.v1.variable_scope('conv_block4'):
      #conv4_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)(maxpool3)
      #conv4_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)(conv4_1)
      #conv4 = _conv_block(maxpool3, ([128, 128]), (3, 3), (1, 1))
      # conv4_emdedding = PatchEmbedding(16* 16, 96*2*2)(maxpool3)
      conv4_1 = SwinTransformer(
      dim=96*2*2,
      num_patch=(16, 16),
      num_heads=12,
      window_size=4,
      shift_size=0,
      num_mlp=96*4*2*2,
      qkv_bias=TRUE,
      dropout_rate=prob,
       )(maxpool3)
      conv4_2 = SwinTransformer(
      dim=96*2*2,
      num_patch=(16, 16),
      num_heads=12,
      window_size=4,
      shift_size=2,
      num_mlp=96*4*2*2,
      qkv_bias=TRUE,
      dropout_rate=prob,
      )(conv4_1)
      # conv4_1 = tf.compat.v1.layers.Conv2D(filters=96*2*2, kernel_size=3, padding='same',activation=tf.nn.relu)(maxpool3)
      # conv4_2 = tf.compat.v1.layers.Conv2D(filters=96*2*2, kernel_size=3, padding='same',activation=tf.nn.relu)(conv4_1)
      conv1_r16 = tf.image.resize(conv1_2, [16, 16])
      conv4_3=tf.reshape(conv4_2,[1,16,16,-1])
      conv4 = tf.compat.v1.layers.Conv2D(filters=96*2*2, kernel_size=3, padding='same',activation=tf.nn.relu)(conv4_3)
      feature.append(tf.concat([conv4, conv1_r16], 3))
      #eature.append(conv4_2)
    
    return feature

def cost_volume(c1, warp, search_range):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(tensor=warp, paddings=[[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(input=c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(input_tensor=c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1)

    return cost_vol


def _H_model(input1, input2, is_training):
    batch_size = tf.shape(input=input1)[0]
  
    with tf.compat.v1.variable_scope('feature_extract', reuse = None): 
      feature1 = feature_extractor(input1,is_training)
    with tf.compat.v1.variable_scope('feature_extract', reuse = True): # H
      feature2 = feature_extractor(input2,is_training)
      
    # Dropout parameter
    keep_prob = 0.5 if is_training==True else 1.0
    
    
    # Regression Net1
    with tf.compat.v1.variable_scope('Reggression_Net1'): 
      feature1[-1],feature2[-1]= crossNonLocalBlock(feature1[-1],feature2[-1],256,0.7,scope="NonLocalBlock")     
      #search_range = 16
      global_correlation = CCL(tf.nn.l2_normalize(feature1[-1],axis=3), tf.nn.l2_normalize(feature2[-1],axis=3))
      global_correlation=SE_block(global_correlation,r=16)   
      #3-convolution layers
      net1_conv1 = tf.compat.v1.layers.Conv2D(filters=512, kernel_size=3, activation=tf.nn.relu)(global_correlation)
      net1_conv2 = tf.compat.v1.layers.Conv2D(filters=512, kernel_size=3, activation=tf.nn.relu)(net1_conv1)
      net1_conv3 = tf.compat.v1.layers.Conv2D(filters=512, kernel_size=3, activation=tf.nn.relu)(net1_conv2)  
      # Flatten dropout_conv4
      net1_flat =  tf.compat.v1.layers.Flatten()(net1_conv3)
      # Two fully-connected layers
      with tf.compat.v1.variable_scope('net1_fc1'):
         net1_fc1 = tf.compat.v1.layers.Dense(1024, activation='relu')(net1_flat)
         net1_fc1 = tf.compat.v1.layers.Dropout(keep_prob)(net1_fc1)
        # net1_fc1 =  tf.compat.v1.layers.Conv2D(filters=1024, kernel_size=1,activation=tf.nn.relu)(net1_conv3)
      with tf.compat.v1.variable_scope('net1_fc2'):
        net1_fc2 = tf.compat.v1.layers.Dense(8)(net1_fc1) #BATCH_SIZE x 8
        # net1_fc2 = tf.compat.v1.layers.Conv2D(filters=8, kernel_size=1,activation=tf.nn.relu)(net1_fc1)
        # net1_fc2 = tf.reshape(net1_fc2,[1,8])
    net1_f = tf.expand_dims(net1_fc2, [2])
    patch_size = 32.
    H1 = solve_DLT(net1_f/4., patch_size)
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    H1 = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    #feature1[-2],feature2[-2]= crossNonLocalBlock(feature1[-2],feature2[-2],128,0.7, scope="NonLocalBlock") 
    feature2_warp = transform(tf.nn.l2_normalize(feature2[-2],axis=3), H1)
    
    
    # Regression Net2
    with tf.compat.v1.variable_scope('Reggression_Net2'):   
      #search_range = 8
      local_correlation_2 = CCL(tf.nn.l2_normalize(feature1[-2],axis=3), feature2_warp) 
      local_correlation_2=SE_block(local_correlation_2,r=16)   
      #3-convolution layers
      net2_conv1 =  tf.compat.v1.layers.Conv2D(filters=256, kernel_size=3, activation=tf.nn.relu)(local_correlation_2)
      net2_conv2 =  tf.compat.v1.layers.Conv2D(filters=256, kernel_size=3, activation=tf.nn.relu)(net2_conv1)
      net2_conv3 =  tf.compat.v1.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2),activation=tf.nn.relu)(net2_conv2)  
      # Flatten dropout_conv4
      net2_flat =  tf.compat.v1.layers.Flatten()(net2_conv3)
      # Two fully-connected layers
      with tf.compat.v1.variable_scope('net2_fc1'):
         net2_fc1 =  tf.compat.v1.layers.Dense(512, activation='relu')(net2_flat)
         net2_fc1 =  tf.compat.v1.layers.Dropout(keep_prob)(net2_fc1)
        # net2_fc1 =  tf.compat.v1.layers.Conv2D(filters=512, kernel_size=1,activation=tf.nn.relu)(net2_conv3)
      with tf.compat.v1.variable_scope('net2_fc2'):
        net2_fc2 =  tf.compat.v1.layers.Dense(8)(net2_fc1) #BATCH_SIZE x 8
        # net2_fc2 = tf.compat.v1.layers.Conv2D(filters=8, kernel_size=1,activation=tf.nn.relu)(net2_fc1)
        # net2_fc2 = tf.reshape(net2_fc2,[1,8])
    net2_f = tf.expand_dims(net2_fc2, [2])
    patch_size = 64.
    H2 = solve_DLT((net1_f+net2_f)/2., patch_size)
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    H2 = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    #feature1[-3],feature2[-3]= crossNonLocalBlock(feature1[-3],feature2[-3],64,0.9, is_training=True, scope="NonLocalBlock")
    feature3_warp = transform(tf.nn.l2_normalize(feature2[-3],axis=3), H2)
    
    
    # Regression Net3
    with tf.compat.v1.variable_scope('Reggression_Net3'):    
      #search_range = 4
      local_correlation_3 = CCL(tf.nn.l2_normalize(feature1[-3],axis=3), feature3_warp) 
      local_correlation_3=SE_block(local_correlation_3,r=16)    
      #3-convolution layers
      net3_conv1 =  tf.compat.v1.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)(local_correlation_3)
      net3_conv2 =  tf.compat.v1.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2),activation=tf.nn.relu)(net3_conv1)
      net3_conv3 =  tf.compat.v1.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2),activation=tf.nn.relu)(net3_conv2) 
      # Flatten dropout_conv4
      net3_flat =  tf.compat.v1.layers.Flatten()(net3_conv3)
      # Two fully-connected layers
      with tf.compat.v1.variable_scope('net3_fc1'):
        net3_fc1 =  tf.compat.v1.layers.Dense(256, activation='relu')(net3_flat)
        # net3_fc1 =  tf.compat.v1.layers.Conv2D(filters=256, kernel_size=1,activation=tf.nn.relu)(net3_conv3)
        net3_fc1 =  tf.compat.v1.layers.Dropout(keep_prob)(net3_fc1)
      with tf.compat.v1.variable_scope('net3_fc2'):
        net3_fc2 =  tf.compat.v1.layers.Dense(8)(net3_fc1) #BATCH_SIZE x 8
    #     net3_fc2 = tf.compat.v1.layers.Conv2D(filters=8, kernel_size=1,activation=tf.nn.relu)(net3_fc1)
    # net3_f = tf.expand_dims(tf.squeeze(tf.squeeze(net3_fc2,1),1), [2])
    net3_f = tf.expand_dims(net3_fc2, [2])
      
    
    return net1_f, net2_f, net3_f,feature1[-1],feature2[-1]

 
