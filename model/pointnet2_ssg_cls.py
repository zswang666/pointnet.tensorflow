import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_layers
import pointnet_layers

def get_inputs(bsize, n_points):
    """ Get input placeholder for ModelNet40 classification
        Args:
            `bsize` (int): batch size
            `n_points` (int): number of points in a point cloud
        Returns:
            `pointcloud_pl` (tensor): point cloud placeholder
            `labels_pl` (tensor): labels placeholder; expected value ranges from 0~39
    """
    pointcloud_pl = tf.placeholder(tf.float32, shape=(bsize, n_points, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(bsize))
    return pointcloud_pl, labels_pl

def get_model(point_cloud, is_training, data_format='NHWC', bn_decay=None):
    """ PointNet++ model using Single Scale Grouping (SSG) for ModelNet40 classification
        Args:
            `point_cloud` (tensor or placeholder): input point cloud of shape BxNx3
            `is_training` (bool): training or testing, only works while using batchnorm
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
            `bn_decay` (tf op): decay function of batch normalization, e.g. tf.train.exponential_decay
        Returns:
            `outputs` (tensor):
            `end_points` (list of tensor):
        Notes:
            1. length of `nsets_list` and `nsamples_list` should be equal
            2. quick example of `mlps_list`:
                >> mlps_list = [[  64,  64, 128],
                                [ 128, 128, 256],
                                [ 256, 512,1024]]
                there are 3 SA layers where the first SA layer contains
                output channels (out) 64 pointwise MLP (pwMLP) --> out 64 pwMLP 
                --> out 128 pwMLP
            3. `n_sets`, `n_samples` and `radius` of the last SA layer should be 
               set carefully to be equivalent to grouping all.
            4. Example setting for SA parameters:
                >> nsets_list = [512, 128, 1]
                >> nsamples_list = [32, 64, 128]
                >> radius_list = [0.2, 0.4, 100]
                >> mlps_list = [[  64,  64, 128],
                                [ 128, 128, 256],
                                [ 256, 512,1024]]
                This will results in xyz: 16x1024x3 --> 16x512x3 --> 16x128x3 --> 16x1x3
                                     points: None --> 16x512x128 --> 16x1287x256 --> 16x1024
    """
    # input arguments
    end_points = {}
    # B: batch size, N: number of points, C: should be 3(xyz)
    if data_format == 'NHWC':
        B, N, C = point_cloud.get_shape().as_list()
    else:
        B, C, N = point_cloud.get_shape().as_list()

    # SA parameters
    nsets_list = [512, 128, 1]
    nsamples_list = [32, 64, 128]
    radius_list = [0.2, 0.4, 100]
    mlps_list = [[  64,  64, 128],
                 [ 128, 128, 256],
                 [ 256, 512,1024]]

    # Set Abstraction (SA) layers
    xyz = point_cloud
    points = None
    sa_params = zip(nsets_list, nsamples_list, radius_list, mlps_list)
    for i, (n_sets, n_samples, radius, mlps) in enumerate(sa_params):
        xyz, points = pointnet_layers.pointnet_sa_module('sa_{}'.format(i+1), xyz, 
                        points, n_sets, n_samples, radius, mlps, is_training, bn_decay)

    # a series of fully-connected and dropout to get cls outputs
    net = tf.squeeze(points, axis=[1]) # flatten for FCs
    net = tf_layers.fully_connected('fc_1', net, 512, bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout_1')
    net = tf_layers.fully_connected('fc_2', net, 256, bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout_2')
    outputs = tf_layers.fully_connected('fc_3', net, 40, bn=True, bn_decay=bn_decay, is_training=is_training)

    return outputs, end_points

def get_loss(pred, label, end_points):
    """ Classification loss
        Args:
            `pred` (tensor): predicted logits of shape Bx40
            `label` (tensor): ground truth label of shape Bx1
            `end_points` (dict): end points from model
        Returns:
            `cls_loss` (tf op): classification loss
    """
    label = tf.cast(tf.squeeze(label, [1]), tf.int32) # to fit cross-entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    cls_loss = tf.reduce_mean(loss)
    tf.add_to_collection('losses', cls_loss)
    return cls_loss

if __name__=='__main__':
    inputs = tf.zeros((16,1024,3))
    outputs, _  = get_model(inputs, True)
    print(outputs.shape)
    import pdb
    pdb.set_trace()
