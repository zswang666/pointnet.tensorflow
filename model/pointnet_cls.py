import os
import sys
import numpy as np
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_layers
import pointnet_layers

def get_model(point_cloud, is_training, data_format='NHWC', bn_decay=None):
    """ PointNet model for ModelNet40 classification
        Args:
            `point_cloud` (tensor or placeholder):
            `is_training` (bool):
            `data_format` (str):
            `bn_decay` (tf op):
        Returns:
            `outputs` (tensor):
            `end_points` (list of tensor):
        Todo:
            1. transformer
            2. finish doc
            3. options for other symmetric function
    """
    end_points = {}
    # B: batch size, N: number of points, C: should be 3(xyz)
    if data_format == 'NHWC':
        _, N, C = point_cloud.get_shape().as_list()
    else:
        _, C, N = point_cloud.get_shape().as_list()
    # convert point cloud BxNx3 --> BxNx1x3
    net = tf.expand_dims(point_cloud, 2)
    # spatial transformer, K=3
    transform = pointnet_layers.transformer('spatial_T', net, 3, is_training, data_format, bn_decay)
    end_points['spatial_T'] = transform
    net = tf.matmul(tf.squeeze(tf.transpose(net, [0,1,3,2]), 3), transform)
    net = tf.expand_dims(net, 2)
    # a series of pointwiseMLP
    net = pointnet_layers.pointwise_mlp('ptmlp_1', net, 64, is_training, data_format, bn_decay)
    net = pointnet_layers.pointwise_mlp('ptmlp_2', net, 64, is_training, data_format, bn_decay)
    # feature transformer, K = 64
    transform = pointnet_layers.transformer('feature_T', net, 64, is_training, data_format, bn_decay)
    end_points['feature_T'] = transform
    net = tf.matmul(tf.squeeze(tf.transpose(net, [0,1,3,2]), 3), transform)
    net = tf.expand_dims(net, 2)
    # a series of pointwise MLP
    net = pointnet_layers.pointwise_mlp('ptmlp_3', net, 64, is_training, data_format, bn_decay)
    net = pointnet_layers.pointwise_mlp('ptmlp_4', net, 128, is_training, data_format, bn_decay)
    net = pointnet_layers.pointwise_mlp('ptmlp_5', net, 1024, is_training, data_format, bn_decay)
    # symmetric function: max pooling.
    net = tf_layers.max_pool2d('maxpool_1', net, [N,1], data_format=data_format)
    # a series of fully-connected and dropout to get cls outputs
    net = tf.squeeze(net, [1,2]) # flatten for FCs
    net = tf_layers.fully_connected('fc_1', net, 512, bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf.contrib.layers.dropout(net, keep_prob=0.7, is_training=is_training, scope='dropout_1')
    net = tf_layers.fully_connected('fc_2', net, 256, bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf.contrib.layers.dropout(net, keep_prob=0.7, is_training=is_training, scope='dropout_2')
    outputs = tf_layers.fully_connected('fc_3', net, 40, activation_fn=None)
    
    return outputs, end_points

def get_loss(pred, label, end_points, reg_weight=0.001):
    """ Classification loss
        Args:
            `pred` (tensor): predicted logits of shape Bx40
            `label` (tensor): ground truth label of shape Bx1
            `end_points` (dict): end points from model
            `reg_weight` (float): matrix orthonormality regularization weight
        Returns:
            `cls_loss` (tf op): classification loss
    """
    # to fit cross-entropy loss
    label = tf.cast(tf.squeeze(label, [1]), tf.int32)
    # classification loss
    with tf.variable_scope('cls_loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        cls_loss = tf.reduce_mean(loss)
        tf.add_to_collection('losses', cls_loss)
        tf.summary.scalar('classify loss', cls_loss)
    # transform as orthonormal matrix
    with tf.variable_scope('mat_loss'):
        T = end_points['feature_T']
        K = T.get_shape()[1].value
        Tsq = tf.matmul(T, tf.transpose(T, perm=[0,2,1]))
        T_err = Tsq - tf.constant(np.eye(K), dtype=tf.float32)
        T_loss = tf.nn.l2_loss(T_err)
        tf.add_to_collection('losses', T_loss)
        tf.summary.scalar('transform loss', T_loss)
    
    return cls_loss + reg_weight*T_loss