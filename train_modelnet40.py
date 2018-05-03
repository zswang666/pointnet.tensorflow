import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import time
from shutil import copyfile

from utils import utils
from utils import tf_layers
from dataloader.modelnet40h5_dataset import ModelNet40H5Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    ##### data
    parser.add_argument('--bsize', type=int, default=32, help='Batch size')
    parser.add_argument('--npoints', type=int, default=1024, help='Number of points in a point cloud')
    parser.add_argument('--train_file', type=str, help='Training file list',
                        default='~/Desktop/workspace/3D/pointnet2/data/modelnet40_ply_hdf5_2048/train_files.txt')
    parser.add_argument('--test_file', type=str, help='Testing file list',
                        default='~/Desktop/workspace/3D/pointnet2/data/modelnet40_ply_hdf5_2048/test_files.txt')
    ##### model
    parser.add_argument('--model', type=str, default='model.pointnet2_ssg_cls', help='Model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--base_bn', type=float, default=0.5, help='Base batchnorm decay')
    parser.add_argument('--bn_decay_steps', type=int, default=6250, help='Batchnorm decay step')
    parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Batchnorm decay rate')
    parser.add_argument('--end_bn', type=float, default=0.99, help='End batchnorm decay')
    parser.add_argument("--bn_staircase", type=utils.str2bool, default=True, help='Batchnorm decay staircase mode')
    ##### learning rate and optimizer
    parser.add_argument('--base_lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--lr_decay_steps', type=int, default=6250, help='Learning rate decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.7, help='Learning rate decay rate')
    parser.add_argument('--end_lr', type=float, default=0.00001, help='End learning rate')
    parser.add_argument("--lr_staircase", type=utils.str2bool, default=True, help='Learning rate decay staircase mode')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer [adam/momentum]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Parameters for momentum optimizer')
    ##### training process
    parser.add_argument('--work_dir', type=str, help='Working directory') # NOTE: you should always specify this for every experiment
    parser.add_argument('--max_epoch', type=int, default=250, help='Maximum training epoch (0~max_epoch+1)')
    parser.add_argument('--eval_steps', type=int, default=1, help='Evaluate every `eval_steps` epochs')
    parser.add_argument('--save_steps', type=int, default=10, help='Save model every `save_steps` epochs')
    parser.add_argument('--log_steps', type=int, default=10, help='Log every `log_steps` batches') # summary is written for every step

    FLAGS = parser.parse_args()
    assert FLAGS.work_dir is not None, 'Please specify --work_dir'
    train(FLAGS)

def train(FLAGS):
    # config
    misc = dict()
    misc['log_steps'] = FLAGS.log_steps
    utils.validate_dir(FLAGS.work_dir)
    misc['batch_size'] = FLAGS.bsize
    save_dir = utils.validate_dir(os.path.join(FLAGS.work_dir,'ckpt'))
    copyfile(os.path.abspath(__file__), os.path.join(FLAGS.work_dir, __file__))

    # define logger
    if FLAGS.model_path is not None:
        misc['logger'] = utils.Logger(os.path.join(FLAGS.work_dir, 'log.txt'), 'a')
    else:
        misc['logger'] = utils.Logger(os.path.join(FLAGS.work_dir, 'log.txt'))

    # model function
    model_fn = utils.dynamic_import(FLAGS.model)

    # build graph
    ops = dict()
    with tf.device('/gpu:{}'.format(FLAGS.gpu)):
        # define start epoch and global step
        start_epoch = tf.get_variable('start_epoch', [], tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        update_start_ep = tf.assign(start_epoch, start_epoch+1)
        ops['global_step'] = tf.get_variable('global_step', [], tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        # prepare dataset
        misc['logger'].write('Preparing dataset...')
        dataset = ModelNet40H5Dataset(FLAGS.train_file, FLAGS.test_file, FLAGS.bsize, FLAGS.npoints)
        misc['trainset_size'], misc['testset_size'] = dataset.size
        misc['label_map'] = dataset._label_map
        trainset = dataset.trainset
        testset = dataset.testset

        misc['logger'].write('Building model graph...')

        # define input and label
        ops['is_training'] = tf.placeholder(tf.bool, [], name='is_training')
        pointcloud = tf.cond(ops['is_training'], lambda: trainset.pointcloud, lambda: testset.pointcloud)
        ops['label'] = tf.cond(ops['is_training'], lambda: trainset.label, lambda: testset.label)

        # define model
        bn_decay = tf_layers.get_bn_decay(ops['global_step'], FLAGS.base_bn, FLAGS.bn_decay_steps, 
                            FLAGS.bn_decay_rate, FLAGS.end_bn, staircase=FLAGS.bn_staircase)
        ops['pred'], end_points = model_fn.get_model(pointcloud, 
                                                     ops['is_training'],
                                                     bn_decay=bn_decay)

        # define loss
        ops['total_loss'] = model_fn.get_loss(ops['pred'], ops['label'], end_points)
        tf.summary.scalar('total loss', ops['total_loss'])

        # define metrics for training
        final_pred = tf.cast(tf.argmax(ops['pred'], 1), tf.uint8)
        final_label = tf.squeeze(ops['label'], 1)
        ops['accuracy'] = tf.contrib.metrics.accuracy(final_pred, final_label, name='accuracy')
        tf.summary.scalar('training accuracy', ops['accuracy'])

        # define optimizer
        lr = tf_layers.get_lr_expdecay(ops['global_step'], FLAGS.base_lr, FLAGS.lr_decay_steps,
                    FLAGS.lr_decay_rate, FLAGS.end_lr, FLAGS.lr_staircase)
        tf.summary.scalar('learning rate', lr)
        if FLAGS.optim == 'momentum':
            optim = tf.train.MomentumOptimizer(lr, momentum=FLAGS.momentum)
        elif FLAGS.optim == 'adam':
            optim = tf.train.AdamOptimizer(lr)
        else:
            raise ValueError('Invalid optimizer {}'.format(FLAGS.optim))
        ops['train'] = optim.minimize(ops['total_loss'], global_step=ops['global_step'])

        # define saver
        saver = tf.train.Saver(max_to_keep=20)

    # create a session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    tf_config.log_device_placement = False
    sess = tf.Session(config=tf_config)

    # define summary writer, saved log in work_dir/log
    ops['summary'] = tf.summary.merge_all()
    misc['summary_writer'] = tf.summary.FileWriter(os.path.join(FLAGS.work_dir, 'log'), sess.graph)
    
    # initialize variables
    misc['logger'].write('Initialize variables...')
    init = tf.global_variables_initializer()
    sess.run(init)

    # load pretrained model
    if FLAGS.model_path is not None:
        misc['logger'].write('Loading pretrained model from {}'.format(FLAGS.model_path))
        saver.restore(sess, os.path.abspath(FLAGS.model_path))
    _start_epoch = sess.run(start_epoch)

    # start training
    misc['logger'].write('Start training from epoch {}'.format(_start_epoch))
    for epoch in range(_start_epoch, FLAGS.max_epoch+1):
        # print epochX
        misc['logger'].write('============= Epoch#{:3d} ============='.format(epoch))

        # train an epoch
        train_an_epoch(sess, ops, misc)

        # evaluate once in a while
        if epoch%FLAGS.eval_steps==0:
            evaluate_an_epoch(sess, ops, misc)

        # save model once in a while
        if epoch%FLAGS.save_steps==0 and epoch!=0:
            saver.save(sess, os.path.join(save_dir, 'model-ep{}.ckpt'.format(epoch)))

        # update start epoch
        sess.run([update_start_ep])

def train_an_epoch(sess, ops, misc):
    """ Training an epoch
        Args:
            `sess` (tf session): tensorflow session to run training process on
            `ops` (dict): a dictionary containing tensorflow ops to be used
            `misc` (dict): a dictionary containing other stuff
    """
    total_steps = misc['trainset_size']//misc['batch_size'] + 1 # total steps in an epoch
    elapsed_time = 0
    for step in range(total_steps):
        tic = time.time()

        feed_dict = {ops['is_training']: True}
        fetch = [ops['train'], 
                 ops['global_step'],
                 ops['total_loss'],
                 ops['accuracy'],
                 ops['summary']]
        _, global_step, total_loss, accuracy, summary = sess.run(fetch, feed_dict=feed_dict)
        misc['summary_writer'].add_summary(summary, global_step)

        toc = time.time()
        elapsed_time += toc-tic

        if step%misc['log_steps'] == 0:
            elapsed_time /= misc['log_steps']
            misc['logger'].write('local [{}/{}] global [{}] loss: {:f}, acc: {:.2f}, batch time: {:.2f}'.format(step, 
                total_steps, int(global_step), total_loss, accuracy, elapsed_time))
            elapsed_time = 0

def evaluate_an_epoch(sess, ops, misc):
    """ Evaluating an epoch
        Args:
            `sess` (tf session): tensorflow session to run training process on
            `ops` (dict): a dictionary containing tensorflow ops to be used
            `misc` (dict): a dictionary containing other stuff
    """
    total_steps = misc['testset_size']//misc['batch_size'] + 1 # total steps in an epoch
    pred_full = []
    label_full = []
    misc['logger'].write('************* Evaluating *************')
    for step in range(total_steps):
        feed_dict = {ops['is_training']: False}
        fetch = [ops['pred'], ops['label'], ops['global_step']]
        pred, label, global_step = sess.run(fetch, feed_dict=feed_dict)

        pred_full.append(pred)
        label_full.append(label)
    pred_full = np.argmax(np.concatenate(pred_full, 0), 1)
    label_full = np.concatenate(label_full, 0)[:,0]
    # overall accuracy
    acc_summary, acc = tf_layers.tf_accuracy(label_full, pred_full)
    misc['summary_writer'].add_summary(acc_summary, global_step)
    # mean accuracy
    mean_acc_summary, mean_acc = tf_layers.tf_mean_accuracy(label_full, pred_full)
    misc['summary_writer'].add_summary(mean_acc_summary, global_step)
    # precision
    precision_summary, precision = tf_layers.tf_precision(label_full, pred_full)
    misc['summary_writer'].add_summary(precision_summary, global_step)
    # confusion matrix
    cm_summary, _ = tf_layers.tf_confusion_matrix(label_full, pred_full, misc['label_map'])
    misc['summary_writer'].add_summary(cm_summary, global_step)
    # log
    misc['logger'].write('global [{}]\n  accuarcy: {}\n  mean-accuracy: {}\n  precision: {}'\
                .format(global_step, acc, mean_acc, precision))
    misc['logger'].write('**************************************')

if __name__=='__main__':
    main()
