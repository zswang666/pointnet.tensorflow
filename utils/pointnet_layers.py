import numpy as np
import tensorflow as tf

import tf_layers
from tf_grouping.tf_grouping import query_ball_point, group_point
from tf_sampling.tf_sampling import farthest_point_sample, gather_point

def pointwise_mlp(scope, inputs, num_output_channels, is_training, data_format='NHWC', bn_decay=None):
    """ pointwise MLP in PointNet
        Args:
            `scopes` (str): scope of this layer
            `inputs` (tensor): input tensor of shape BxNxWxC
            `num_output_channels` (int): specify output dimensions of the mlp
            `is_training` (bool): training or testing, only works while using batchnorm
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
            `bn_decay` (tf op): decay function of batch normalization
        Returns:
            `outputs` (tensor): output tensor
        Notes:
            1. Pointwise MLP is performed on the 'H' and 'W' axis, so in this two axis, each
               entry must specify a point.
    """
    outputs = tf_layers.conv2d(scope, inputs, num_output_channels, [1,1], stride=[1,1], padding='VALID', 
                    data_format=data_format, bn=True, bn_decay=bn_decay, is_training=is_training)
    
    return outputs

def transformer(scope, inputs, K, is_training, data_format='NHWC', bn_decay=None):
    """ spatial or feature space transformer for PointNet
        Args:
        Returns:
            `transform` (tensor): transformation matrix of shape BxKxK
        Todo:
            1. complete doc
    """
    if data_format == 'NHWC': # BxNx1xC
        N = inputs.get_shape()[1].value
    else: # BxCxNx1
        N = inputs.get_shape()[2].value
    with tf.variable_scope(scope):
        # a series of pointwise MLP
        net = pointwise_mlp('ptmlp_1', inputs, 64, is_training, data_format, bn_decay)
        net = pointwise_mlp('ptmlp_2', net, 128, is_training, data_format, bn_decay)
        net = pointwise_mlp('ptmlp_3', net, 1024, is_training, data_format, bn_decay)
        # symmetric function: max pooling
        net = tf_layers.max_pool2d('maxpool_1', net, [N,1], data_format=data_format)
        # a series of fully connected
        net = tf.squeeze(net, [1,2])
        net = tf_layers.fully_connected('fc_1', net, 512, bn=True, bn_decay=bn_decay, is_training=is_training)
        net = tf_layers.fully_connected('fc_2', net, 256, bn=True, bn_decay=bn_decay, is_training=is_training)
        # transformer
        with tf.variable_scope('transform'):
            weights = tf.get_variable('weights', [256, K*K],
                                      initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [K*K], # initialize as eye(K)
                                     initializer=tf.constant_initializer(np.eye(K).flatten()),
                                     dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)
            transform = tf.reshape(transform, [-1, K, K])
        
    return transform
    

def sample_and_group(xyz, points, n_sets, n_samples, radius):
    """ Perform sampling and grouping for set abstraction
        Args:
            `xyz` (tensor): xyz of the point cloud; shape BxNx3
            `points` (tensor): features of all points, if 'None', use `xyz` as points; shape BxNxC
            `n_sets` (int): number of sets
            `n_samples` (int): number of samples in a set
            `radius` (float): radius for query-ball-point sampling
        Returns:
            `sampled_xyz` (tensor): the centers of all abstracted set in xyz space; shape Bx`n_sets`x3
            `grouped_xyz` (tensor): grouped points in xyz space normalized in local regions; 
                                    shape Bx`n_sets`x`n_samples`x3
            `grouped_points` (tensor): grouped points in feature space; shape Bx`n_sets`x`n_samples`xC
        Notes:
            1. supported sampling methods
                - farthest point sampling
            2. supported grouping methods
                - query ball point grouping
        Todo:
            1. add KNN grouping
            2. `use_xyz`
    """
    # sample representative points with `xyz`
    fp_idx = farthest_point_sample(n_sets, xyz)
    sampled_xyz = gather_point(xyz, fp_idx)
    # group neighboring points with centers as `sampled_xyz` in xyz space
    group_idx, pts_cnt = query_ball_point(radius, n_samples, xyz, sampled_xyz)
    grouped_xyz = group_point(xyz, group_idx)
    # translation normalization (each set centered at `sampled_xyz`)
    grouped_xyz -= tf.tile(tf.expand_dims(sampled_xyz, 2), [1,1,n_samples,1])
    # use the same grouping index in xyz space to group feature points
    if points is not None:
        grouped_points = group_point(points, group_idx)
    else:
        grouped_points = grouped_xyz
    
    return sampled_xyz, grouped_xyz, grouped_points

def pointnet_sa_module(scope, xyz, points, n_sets, n_samples, radius, mlp, is_training, bn_decay=None):
    """ PointNet Set Abstraction (SA) module
        Args:
            `scope` (str): scope of this layer
            `xyz` (tensor): xyz of the point cloud; shape BxNx3
            `points` (tensor): features of all points, if 'None', use `xyz` as points; shape BxNxC
            `n_sets` (int): number of sets
            `n_samples` (int): number of samples in a set
            `radius` (float): radius for query-ball-point sampling
            `mlp` (list or tuple of int): output sizes of the series of pointwise MLP
            `is_training` (bool): training or testing, only works while using batchnorm
            `bn_decay` (tf op): decay function of batch normalization for pointwise MLP
        Returns:
            `new_xyz` (tensor): new set of xyz after set abstraction; shape Bx`n_sets`x3
            `new_points` (tensor): new set of feature points after set abstraction; shape Bx`n_sets`xC
        Todo:
            1. data format 'NCHW'
            2. different pooling methods
    """
    with tf.variable_scope(scope):
        # sampling and grouping
        new_xyz, _, new_points = sample_and_group(xyz, points, n_sets, n_samples, radius)

        # pointwise MLPs
        for i, num_output_channels in enumerate(mlp):
            new_points = pointwise_mlp('pwmlp_{}'.format(i+1), new_points, num_output_channels,
                                       is_training, bn_decay=bn_decay)

        # pooling in local region
        new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='max-pool')

        # squeeze extra dimension
        new_points = tf.squeeze(new_points, axis=[2])

    return new_xyz, new_points