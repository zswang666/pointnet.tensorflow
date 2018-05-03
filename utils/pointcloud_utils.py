import sys
import numpy as np
import tensorflow as tf
from functools import partial

# TODO: tf_rotate_perturb_point_cloud

def tf_rotation_matrix_x(theta):
    """ Tensorflow op: create rotation matrix w.r.t. X axis (theta in [0,2*pi]) """
    cos, sin = tf.cos(theta), tf.sin(theta)
    rot = tf.stack([1,   0,      0,
                    0, cos, -1*sin,
                    0, sin,    cos])
    rot = tf.reshape(rot, (3,3))
    return rot

def tf_rotation_matrix_y(theta):
    """ Tensorflow op: create rotation matrix w.r.t. Y axis (theta in [0,2*pi]) """
    cos, sin = tf.cos(theta), tf.sin(theta)
    rot = tf.stack([   cos,  0, sin,
                         0,  1,   0,
                    -1*sin,  0, cos])
    rot = tf.reshape(rot, (3,3))
    return rot

def tf_rotation_matrix_z(theta):
    """ Tensorflow op: create rotation matrix w.r.t. Z axis (theta in [0,2*pi]) """
    cos, sin = tf.cos(theta), tf.sin(theta)
    rot = tf.stack([cos, -1*sin, 0,
                    sin,    cos, 0,
                      0,      0, 1])
    rot = tf.reshape(rot, (3,3))
    return rot

def tf_random_rotate_point_cloud(pc):
    """ Tensorflow op: randomly rotate a point cloud (Nx3) w.r.t. Y axes """
    theta = tf.random_uniform([3], 0., 1.) * 2 * np.pi
    rot_y = tf_rotation_matrix_y(theta[1])
    rotated_pc = tf.matmul(pc, rot_y)
    return rotated_pc

def tf_random_rotateXYZ_point_cloud(pc):
    """ Tensorflow op: randomly rotate a point cloud (Nx3) w.r.t. X->Y->Z axis """
    theta = tf.random_uniform([3], 0., 1.) * 2 * np.pi
    rot_x = tf_rotation_matrix_x(theta[0])
    rot_y = tf_rotation_matrix_y(theta[1])
    rot_z = tf_rotation_matrix_z(theta[2])
    rotated_pc = tf.matmul(tf.matmul(tf.matmul(pc, rot_x), rot_y), rot_z)
    return rotated_pc

def tf_rotate_perturb_point_cloud():
    """ Tensorflow op: perturbation by slightly rotating a point cloud """
    raise NotImplementedError

def tf_random_scale_point_cloud(pc, low, high):
    """ Tensorflow op: randomly scale a point cloud (Nx3) """
    scale = tf.random_uniform([], low, high)
    return scale * pc

def tf_random_shift_point_cloud(pc, shift_range):
    """ Tensorflow op: randomly shift a point cloud (Nx3) w.r.t. X,Y,Z axes """
    shifts = tf.random_uniform([3], -shift_range, shift_range)
    return pc + shifts

def tf_jitter_point_cloud(pc, sigma, clip):
    """ Tensorflow op: jitter a point cloud.
        Notes:
            1. jitter is performed pointwise
            2. jitter follows gaussian distribution and will then be clipped.
    """
    jittered_val = tf.random_normal(pc.shape, 0, sigma)
    jittered_val = tf.clip_by_value(jittered_val, -clip, clip)
    return pc + jittered_val

def setup_pcl_viewer():
    """ setup a point cloud viewer using vispy and return a drawing function """
    import vispy.scene
    from vispy.scene import visuals

    # make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    # create scatter object and fill in the data
    init_pc = np.random.normal(size=(100, 3), scale=0.2)
    scatter = visuals.Markers()
    draw_fn = partial(scatter.set_data, edge_color=None, face_color=(1, 1, 1, .5), size=5)
    draw_fn(init_pc)
    view.add(scatter)
    # set camera
    view.camera = 'turntable' # ['turntable','arcball']
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    
    return draw_fn
