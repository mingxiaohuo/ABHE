import os
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from models import H_estimator, disjoint_augment_image_pair
from loss_functions import intensity_loss
from utils import load, save, DataLoader
import constant
import numpy as np


#os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#,8,9,10,11,12,13,14,15
gpus=[0]
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# gpus=[0,1,2]


train_folder = constant.TRAIN_FOLDER
test_folder = constant.TEST_FOLDER

batch_size = constant.TRAIN_BATCH_SIZE
iterations = constant.ITERATIONS

height, width = 128, 128

summary_dir = constant.SUMMARY_DIR
snapshot_dir = constant.SNAPSHOT_DIR

def average_gradients(tower_grads):
    average_grads = []
    ##grad_and_vars：((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
 
# define dataset
with tf.compat.v1.name_scope('dataset'):
    ##########training###############
    ###input###
    train_data_loader = DataLoader(train_folder)
    train_data_dataset = train_data_loader(batch_size=batch_size)
    train_data_it = tf.compat.v1.data.make_one_shot_iterator(train_data_dataset)
    (train_input_tensor, train_size_tensor) = train_data_it.get_next()
    train_input_tensor.set_shape([batch_size, height, width, 3*2])
    train_size_tensor.set_shape([batch_size, 2, 1])
    train_inputs = train_input_tensor
    train_size = train_size_tensor
    #print('train inputs = {}'.format(train_inputs))
    #only training dataset augment
#with tf.compat.v1.name_scope('disjoint_augment'):
    train_inputs_aug = disjoint_augment_image_pair(train_inputs)
    train_inputs_splits = tf.split(train_inputs, num_or_size_splits=len(gpus), axis=0)
    train_inputs_aug_splits = tf.split(train_inputs_aug, num_or_size_splits=len(gpus), axis=0)
 
g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
g_lrate =  tf.compat.v1.train.exponential_decay(0.0001, g_step, decay_steps=50000/4, decay_rate=0.96)
g_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
tower_grads = []  
tower_loss = []   

for i in range(len(gpus)): 
      with tf.compat.v1.device('/gpu:%d' % i): 
         with tf.compat.v1.variable_scope('generator',reuse=tf.compat.v1.AUTO_REUSE):
                   
# define training generator function
#with tf.compat.v1.variable_scope('generator', reuse=None):
           print('training = {}'.format(tf.compat.v1.get_variable_scope().name))
           #train_net1_f, train_net2_f, train_net3_f, train_warp2_H1, train_warp2_H2, train_warp2_H3, train_one_warp_H1, train_one_warp_H2, train_one_warp_H3= H_estimator(train_inputs_aug_splits[i], train_inputs_splits[i], True)
           #train_net1_f, train_net2_f, train_net3_f, train_warp2_H1, train_warp2_H2, train_warp2_H3, train_one_warp_H1, train_one_warp_H2, train_one_warp_H3,train_one_1_warp_H1,train_one_2_warp_H2,train_feature11,train_feature2_H1,train_feature12,train_feature2_H2 = H_estimator(train_inputs_aug_splits[i], train_inputs_splits[i], True)
           train_net1_f, train_net2_f, train_net3_f, train_warp2_H1, train_warp2_H2, train_warp2_H3, train_one_warp_H1, train_one_warp_H2, train_one_warp_H3,train_one_1_warp_H1,train_feature11,train_feature2_H1 = H_estimator(train_inputs_aug_splits[i], train_inputs_splits[i], True)
#with tf.compat.v1.name_scope('loss'):

           lam_lp = 1
           loss1 = intensity_loss(gen_frames=train_warp2_H1, gt_frames=train_inputs_splits[i][...,0:3]*train_one_warp_H1, l_num=1)
           loss2 = intensity_loss(gen_frames=train_warp2_H2, gt_frames=train_inputs_splits[i][...,0:3]*train_one_warp_H2, l_num=1)
           loss3 = intensity_loss(gen_frames=train_warp2_H3, gt_frames=train_inputs_splits[i][...,0:3]*train_one_warp_H3, l_num=1)
           lp_loss = 16. * loss1 + 4. * loss2 + 1. * loss3
           content_lp=5
           loss4 = intensity_loss(gen_frames=train_feature2_H1, gt_frames=train_feature11*train_one_1_warp_H1, l_num=1)
           #loss5 = intensity_loss(gen_frames=train_feature2_H2, gt_frames=train_feature12*train_one_2_warp_H2, l_num=1)
           content_loss = 4. *loss4  
#with tf.compat.v1.name_scope('training'):
           g_loss = tf.add_n([lp_loss * lam_lp,content_lp*content_loss], name='g_loss')
           #g_loss = tf.add_n([lp_loss * lam_lp],name='g_loss')
           tf.compat.v1.summary.scalar('g_loss_gpu%d' % i, g_loss)
        #    tf.compat.v1.summary.image('train_feature12%d' % i, train_feature12)
        #    tf.compat.v1.summary.image('train_feature2_H2%d' % i,train_feature2_H2)
        #    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
        #    g_lrate =  tf.compat.v1.train.exponential_decay(0.0001, g_step, decay_steps=50000/4, decay_rate=0.96)
        #    g_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
           g_vars = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
           #tf.compat.v1.get_variable_scope().reuse_variables()
           grads = g_optimizer.compute_gradients(g_loss,var_list=g_vars)
           #tf.compat.v1.get_variable_scope().reuse_variables()
           #grads = g_optimizer.compute_gradients(g_loss,var_list=tf.compat.v1.trainable_variables())#var_list=g_vars
        #    for i, (g, v) in enumerate(grads):
        #         if g is not None:
        #           grads[i] = (tf.clip_by_norm(g, 3), v)  # clip gradients
           tower_grads.append([x for x in grads if x[0] is not None])
           tower_loss.append(g_loss)
           
           #g_train_op = g_optimizer.apply_gradients(grads, global_step=g_step, name='g_train_op')
    
avg_tower_loss = tf.reduce_mean(tower_loss, axis=0)
tf.compat.v1.summary.scalar('avg_tower_loss', avg_tower_loss)
grads_avg = average_gradients(tower_grads)
# add all to summaries
# tf.compat.v1.summary.scalar(tensor=g_loss, name='g_loss')
# tf.compat.v1.summary.scalar(tensor=loss1, name='loss1')
# tf.compat.v1.summary.scalar(tensor=loss2, name='loss2')
# tf.compat.v1.summary.scalar(tensor=loss3, name='loss3')

# tf.compat.v1.summary.image(tensor=train_inputs[...,0:3], name='train_inpu1')
# tf.compat.v1.summary.image(tensor=train_inputs[...,3:6], name='train_inpu2')
# tf.compat.v1.summary.image(tensor=train_warp2_H3, name='train_warp2_H3')

summary_op = tf.compat.v1.summary.merge_all()
update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update):  
        g_train_op = g_optimizer.apply_gradients(grads_avg, global_step=g_step)
 
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)#allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    # summaries
    summary_writer = tf.compat.v1.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.compat.v1.global_variables_initializer())
    print('Init successfully!')

    # tf saver
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=None)
    restore_var = [v for v in tf.compat.v1.global_variables()]
    loader = tf.compat.v1.train.Saver(var_list=restore_var)
    print("snapshot_dir")
    print(snapshot_dir)
    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None

    print("============starting training===========")
    while _step < iterations:
        try:
            print('Training generator...')
           # _, _g_lr, _step, _lp_loss,_content_loss, _g_loss, _summaries = sess.run([g_train_op, g_lrate, g_step, lp_loss,content_loss,g_loss, summary_op])
            _, _g_lr, _step,_avg_tower_loss, _summaries = sess.run([g_train_op, g_lrate, g_step,avg_tower_loss,summary_op])
            if _step % 100 == 0:
            #if _step % 1 == 0:
                print('GeneratorModel : Step {}, lr = {:.8f}'.format(_step, _g_lr))
                print('                 Global      Loss : ', _avg_tower_loss)
                # print('                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_lp_loss, lam_lp, _lp_loss * lam_lp))
                # print('                 content   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_content_loss, content_lp, _content_loss * content_lp))
            if _step % 1000 == 0:
                summary_writer.add_summary(_summaries, global_step=_step)
                print('Save summaries...')

            if _step % 20000 == 0:
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            print('Finish successfully!')
            save(saver, sess, snapshot_dir, _step)
            break