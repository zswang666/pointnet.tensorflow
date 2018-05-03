import tensorflow as tf
import numpy as np

# TODO: conv3d, max_pool3d, avg_pool3d, dropout

def _variable_on_cpu(name, shape, initializer, id=0, dtype=tf.float32):
    """ Create a tensorflow variable on CPU
        Args:
            `name` (str): name of the variable
            `shape` (list or tuple): shape of the variable
            `initializer` (tf function): initializer of the variable
            `id` (int): CPU id
            `dtype` (tf dtype): data type of the varible; default to
        Returns:
            `var` (tensor): variable tensor
    """
    with tf.device('/cpu:{}'.format(id)):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_on_gpu(name, shape, initializer, id=0, dtype=tf.float32):
    """ Create a tensorflow variable on GPU
        Args:
            `name` (str): name of the variable
            `shape` (list or tuple of int): shape of the variable
            `initializer` (tf function): initializer of the variable
            `id` (int): GPU id
            `dtype` (tf dtype): data type of the varible; default to
        Returns:
            `var` (tensor): variable tensor
        Notes:
            - possible `initializer` may be:
                tf.constant_initializer(value)
                tf.contrib.layers.xavier_initializer()
                tf.truncated_normal_initializer(stddev=stddev)
    """
    with tf.device('/gpu:{}'.format(id)):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

_get_variable = _variable_on_gpu

def _variable_with_weight_decay(name, shape, wd, initializer, id=0, dtype=tf.float32):
    """ Create a tensorflow variable with weight decay on GPU 
        Args:
            `name` (str): name of the variable
            `shape` (list or tuple of int): shape of the variable
            `wd` (float): coefficient of weight decay
            `initializer` (tf function instance): initializer of the variable
            `id` (int): GPU id
            `dtype` (tf dtype): data type of the varible; default to
        Returns:
            `var` (tensor): variable tensor
        Notes:
            - weight decay loss is added to collection 'losses' with name 'weight_loss'
    """
    var = _get_variable(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def batch_norm(scope, inputs, is_training, bn_decay, data_format='NHWC'):
    """ tensorflow layer: Batch normalization
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): input tensor
            `is_training` (bool): training or testing
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
        Returns:
            `outputs` (tensor): output tensor
    """
    bn_decay = bn_decay if bn_decay is not None else 0.9
    outputs = tf.contrib.layers.batch_norm(inputs, center=True, scale=True,
                                           is_training=is_training, decay=bn_decay,
                                           updates_collections=None, scope=scope, 
                                           data_format=data_format)
    return outputs

def conv1d(scope, inputs, num_output_channels, kernel_size, stride=1, padding='SAME',
           data_format='NHWC', initializer=tf.contrib.layers.xavier_initializer(), weight_decay=None, 
           activation_fn=tf.nn.relu, bn=False, bn_decay=None, is_training=None):
    """ tensorflow layer: 1D convolution with non-linear activation
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 3D input tensor of shape BxLxC or BxCxL
            `num_output_channels` (int): number of output channels
            `kernel_size` (int): kernel size of 1D convolution
            `stride` (int): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
            `weight_decay` (float): weight decay of trainable variables
            `activation_fn` (tf layer): activation function; default to 'tf.nn.relu',
                                        set to 'None' for no activation function
            `bn` (bool): whether use batch normalization
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `is_training` (bool): training or testing, only works while using batchnorm
        Returns:
            `outputs` (tensor): output tensor
        Notes:
            1. not tested
    """
    # check arguments
    assert data_format=='NHWC' or data_format=='NCHW', 'Invalid data format {}'.format(data_format)
    with tf.variable_scope(scope):
        # specify weights and biases
        if data_format == 'NHWC':
            B, H, W, C = inputs.get_shape().as_list()
        else:
            B, C, H, W = inputs.get_shape().as_list()
        kernel_shape = [kernel_size, C, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             wd=weight_decay,
                                             initializer=initializer)
        biases = _get_variable('biases', [num_output_channels],
                               tf.constant_initializer(0.0))
        # perform 1D convolution
        outputs = tf.nn.conv1d(inputs, kernel, stride=stride, padding=padding, data_format=data_format)
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)
        # perform batch normalization if set
        if bn:
            outputs = batch_norm('bn', outputs, is_training, bn_decay=bn_decay, data_format=data_format)
        # perform activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def conv2d(scope, inputs, num_output_channels, kernel_size, stride=[1,1], padding='SAME',
           data_format='NHWC', initializer=tf.contrib.layers.xavier_initializer(), weight_decay=None, 
           activation_fn=tf.nn.relu, bn=False, bn_decay=None, is_training=None):
    """ tensorflow layer: 2D convolution with non-linear activation
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 4D input tensor of shape BxHxWxC or BxCxHxW
            `num_output_channels` (int): number of output channels
            `kernel_size` (list or tuple of int; len=2): kernel size of 1D convolution
            `stride` (list or tuple of int; len=2): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
            `weight_decay` (float): weight decay of trainable variables
            `activation_fn` (tf layer): activation function; default to 'tf.nn.relu',
                                        set to 'None' for no activation function
            `bn` (bool): whether use batch normalization
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `is_training` (bool): training or testing, only works while using batchnorm
        Returns:
            `outputs` (tensor): output tensor
    """
    # check arguments
    assert data_format=='NHWC' or data_format=='NCHW', 'Invalid data format {}'.format(data_format)
    with tf.variable_scope(scope):
        # specify weights and biases
        if data_format == 'NHWC':
            B, H, W, C = inputs.get_shape().as_list()
        else:
            B, C, H, W = inputs.get_shape().as_list()
        kernel_h, kernel_w = kernel_size
        kernel_shape = [kernel_h, kernel_w, C, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             wd=weight_decay,
                                             initializer=initializer)
        biases = _get_variable('biases', [num_output_channels],
                               tf.constant_initializer(0.0))
        # perform 2D convolution
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel, strides=[1, stride_h, stride_w, 1], padding=padding, data_format=data_format)
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)
        # perform batch normalization if set
        if bn:
            outputs = batch_norm('bn', outputs, is_training, bn_decay=bn_decay, data_format=data_format)
        # perform activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def conv2d_transpose(scope, inputs, num_output_channels, kernel_size, stride=[1,1], padding='SAME',
                     data_format='NHWC', initializer=tf.contrib.layers.xavier_initializer(), weight_decay=None, 
                     activation_fn=tf.nn.relu, bn=False, bn_decay=None, is_training=None):
    """ tensorflow layer: 2D convolution transpose with non-linear activation
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 4D input tensor of shape BxHxWxC or BxCxHxW
            `num_output_channels` (int): number of output channels
            `kernel_size` (list or tuple of int; len=2): kernel size of 1D convolution
            `stride` (list or tuple of int; len=2): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
            `weight_decay` (float): weight decay of trainable variables
            `activation_fn` (tf layer): activation function; default to 'tf.nn.relu',
                                        set to 'None' for no activation function
            `bn` (bool): whether use batch normalization
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `is_training` (bool): training or testing, only works while using batchnorm
        Returns:
            `outputs` (tensor): output tensor
        Notes:
            1. not tested
    """
    # from slim.convolution2d_transpose
    def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
        dim_size *= stride_size
        if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
        return dim_size
    # check arguments
    assert data_format=='NHWC' or data_format=='NCHW', 'Invalid data format {}'.format(data_format)
    with tf.variable_scope(scope):
        # specify weights and biases
        if data_format == 'NHWC':
            B, H, W, C = inputs.get_shape().as_list()
        else:
            B, C, H, W = inputs.get_shape().as_list()
        kernel_h, kernel_w = kernel_size
        kernel_shape = [kernel_h, kernel_w, num_output_channels, C] # in/out channel reverse comparing to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             wd=weight_decay,
                                             initializer=initializer)
        biases = _get_variable('biases', [num_output_channels],
                               tf.constant_initializer(0.0))
        # perform 2D convolution transpose
        stride_h, stride_w = stride
        outH = get_deconv_dim(H, stride_h, kernel_h, padding)
        outW = get_deconv_dim(W, stride_w, kernel_w, padding)
        if data_format == 'NHWC':
            output_shape = [B, outH, outW, num_output_channels]
        else:
            output_shape = [B, num_output_channels, outH, outW]
        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                                         strides=[1, stride_h, stride_w, 1], 
                                         padding=padding, data_format=data_format)
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)
        # perform batch normalization if set
        if bn:
            outputs = batch_norm('bn', outputs, is_training, bn_decay=bn_decay, data_format=data_format)
        # perform activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def fully_connected(scope, inputs, num_outputs, initializer=tf.contrib.layers.xavier_initializer(), 
                    weight_decay=None, activation_fn=tf.nn.relu, bn=False, bn_decay=None, is_training=None):
    """ tensorflow layer: fully connected with non-linear activation
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 2D input tensor of shape BxHxWxC or BxN
            `num_outputs` (int): number of output dimensions
            `weight_decay` (float): weight decay of trainable variables
            `activation_fn` (tf layer): activation function; default to 'tf.nn.relu',
                                        set to 'None' for no activation function
            `bn` (bool): whether use batch normalization
            `bn_decay` (tf op or float): decay function of batch normalization, e.g. tf.train.exponential_decay
            `is_training` (bool): training or testing, only works while using batchnorm
        Returns:
            `outputs` (tensor): output tensor
    """
    with tf.variable_scope(scope):
        # specify weights and biases
        N = inputs.get_shape()[-1].value
        weight = _variable_with_weight_decay('weights',
                                             shape=[N, num_outputs],
                                             wd=weight_decay,
                                             initializer=initializer)
        biases = _get_variable('biases', [num_outputs],
                               tf.constant_initializer(0.0))
        # perform fully connected layer
        outputs = tf.nn.bias_add(tf.matmul(inputs, weight), biases)
        # perform batch normalization if set
        if bn:
            outputs = batch_norm('bn', outputs, is_training, bn_decay=bn_decay)
        # perform activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def max_pool2d(scope, inputs, kernel_size, stride=[2,2], padding='VALID', data_format='NHWC'):
    """ tensorflow layer: 2D max pooling
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 4D input tensor of shape BxHxWxC or BxCxHxW
            `num_output_channels` (int): number of output channels
            `kernel_size` (list or tuple of int; len=2): kernel size of 1D convolution
            `stride` (list or tuple of int; len=2): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
        Returns:
            `outputs` (tensor): output tensor
    """
    with tf.variable_scope(scope):
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name='max-pool2d',
                                 data_format=data_format)
    return outputs

def avg_pool2d(scope, inputs, kernel_size, stride=[2,2], padding='VALID', data_format='NHWC'):
    """ tensorflow layer: 2D average pooling
        Args:
            `scope` (str): scope of this layer
            `inputs` (tensor): 4D input tensor of shape BxHxWxC or BxCxHxW
            `num_output_channels` (int): number of output channels
            `kernel_size` (list or tuple of int; len=2): kernel size of 1D convolution
            `stride` (list or tuple of int; len=2): stride of convolution
            `padding` (str): padding of convolution; 'SAME' or 'VALID'
            `data_format` (str): data format of inputs; 'NHWC' or 'NCHW'
        Returns:
            `outputs` (tensor): output tensor
        Notes:
            1. not tested
    """
    with tf.variable_scope(scope):
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name='avg-pool2d',
                                 data_format=data_format)
    return outputs

def get_lr_expdecay(step, base_lr, decay_steps, decay_rate, end_lr, staircase=False):
    """ get learning rate decay tf operation
        Args:
            `step` (tensor): global step
            `base_lr` (float): base learning rate
            `decay_steps` (int): decay steps
            `decay_rate` (float): decay rate
            `end_lr` (float): end learning rate
            `staircase` (bool): params of exponential decay
        Returns:
            `lr` (tensor): learning rate op
        Notes:
            1. lr = max(base_lr * decay_rate^(step / decay_steps), end_lr)
    """
    lr = tf.train.exponential_decay(
            base_lr,
            step,
            decay_steps,
            decay_rate,
            staircase=staircase)
    lr = tf.maximum(lr, end_lr) # clipping to end learning rate
    return lr

def get_bn_decay(step, base_bn, decay_step, decay_rate, end_bn, staircase=False):
    """ get batch normalization decay tf operation """
    bn_momentum = tf.train.exponential_decay(
                        base_bn,
                        step,
                        decay_step,
                        decay_rate,
                        staircase=staircase)
    bn_decay = tf.minimum(end_bn, 1 - bn_momentum)
    return bn_decay

def tf_accuracy(correct_labels, predict_labels, tensor_name='accuracy'):
    """ compute accuracy and return a tensorflow summary
        Args:
            `correct_labels` : true classification categories
            `predict_labels` : predicted classification categories
            `tensor_name` (str): Name for the output summay tensor
        Returns:
            `summary` (tf summary): TensorFlow summary
            `acc` (float): accuracy
    """
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(correct_labels, predict_labels)
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=tensor_name, simple_value=acc), 
    ])
    return summary, acc

def tf_mean_accuracy(correct_labels, predict_labels, tensor_name='mean accuracy'):
    """ compute mean (average class) accuracy and return a tensorflow summary
        Args:
            `correct_labels` : true classification categories
            `predict_labels` : predicted classification categories
            `tensor_name` (str): Name for the output summay tensor
        Returns:
            `summary` (tf summary): TensorFlow summary
            `mean_acc` (float): mean accuracy
    """
    from sklearn.metrics import accuracy_score
    labels = np.unique(correct_labels)
    mean_acc = []
    for l in labels:
        valid_idx = (correct_labels==l)
        gt = correct_labels[valid_idx]
        pd = predict_labels[valid_idx]
        mean_acc.append(accuracy_score(gt, pd))
    mean_acc = np.mean(mean_acc)
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=tensor_name, simple_value=mean_acc),
    ])
    return summary, mean_acc

def tf_precision(correct_labels, predict_labels, tensor_name='precision', average='macro'):
    """ compute precision and return a tensorflow summary
        Args:
            `correct_labels` : true classification categories
            `predict_labels` : predicted classification categories
            `tensor_name` (str): Name for the output summay tensor
            `average` (str): 'precision_score' params, 
                    check http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html 
        Returns:
            `summary` (tf summary): TensorFlow summary
            `p` (float): precision
        Notes:
            1. average='macro' is to calculate metrics for each label, and find their unweighted mean
    """
    from sklearn.metrics import precision_score
    p = precision_score(correct_labels, predict_labels, average=average)
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=tensor_name, simple_value=p), 
    ])
    return summary, p

def tf_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion Matrix', tensor_name='confusion matrix', normalize=False):
    ''' plot confusion matrix and return a tensorflow summary
        see https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard 
        Args:
            `correct_labels` : true classification categories
            `predict_labels` : predicted classification categories
            `labels` : a list of labels which will be used to display the axix labels
            `title` (str): Title for the matrix
            `tensor_name` (str): Name for the output summay tensor
        Returns:
            `summary` (tf summary): TensorFlow summary
            `cm` : 
        Notes:
            1. Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
            2. Currently, some of the ticks dont line up due to rotations.
        Todo:
            1. complete doc
    '''
    import re
    from textwrap import wrap
    import itertools
    import matplotlib
    import tfplot
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)

    return summary, cm