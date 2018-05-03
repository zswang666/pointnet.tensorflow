import os
import sys
import numpy as np
import tensorflow as tf
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from utils import utils
from utils import pointcloud_utils

def _parse_datafile(f):
    """ parse date list file, e.g. train_files.txt """
    f = os.path.abspath(os.path.expanduser(f))
    root_dir = os.path.join('/',*f.split('/')[:-3]) # should be -1
    data_list = []
    for line in open(f):
        data_list.append(os.path.join(root_dir, line.rstrip()))
    return data_list

def label_map():
    """ int-string label mapping of ModelNet40 """
    return ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
            'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
            'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 
            'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
            'wardrobe', 'xbox']

class ModelNet40H5Dataset(object):
    def __init__(self, train_datafile, test_datafile, batch_size, n_points, shuffle=False):
        """ ModelNet40 .h5 file dataset
            Args:
                `train_datafile` (str): file containing a list of .h5 filename for training
                `test_datafile` (str): file containing a list of .h5 filename for testing
                `batch_size` (int): batch size of the dataset
                `n_points` (int): number of points in a point cloud
                `shuffle` (bool): whether to shuffle dataset
            Notes:
                1. output batch data is
                    'pc': shape=(batch_size, n_points, 3); dtype=tf.float32
                    'label': shape=(batch_size,); dtype=tf.uint8
        """
        # parameters
        train_data_list = _parse_datafile(train_datafile)
        test_data_list = _parse_datafile(test_datafile)
        self.batch_size = batch_size
        self.n_points = n_points
        self.shuffle = shuffle

        # load all data at a time
        self._label_map = label_map()
        self.train_data, self.train_label, self.trainset_size = self._load_data(train_data_list)
        self.test_data, self.test_label, self.testset_size = self._load_data(test_data_list)

        # create dataset
        self._trainset = self._create(self.train_data, self.train_label, self.trainset_size, True, True)
        self._testset = self._create(self.test_data, self.test_label, self.testset_size, False, False)

    def _create(self, data, label, dataset_size, shuffle, is_training):
        """ create tensorflow dataset """
        data_format = collections.namedtuple('Data', 'pointcloud, label')

        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=dataset_size)
        if is_training:
            dataset = dataset.map(self._process, num_parallel_calls=10)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_data, batch_label = iterator.get_next()

        return data_format(pointcloud=batch_data, label=batch_label)

    def _process(self, pc, label):
        """ data augmentation for each point cloud """
        pc = pointcloud_utils.tf_random_rotate_point_cloud(pc)
        #pc = pointcloud_utils.tf_random_scale_point_cloud(pc, 0.8, 1.25)
        #pc = pointcloud_utils.tf_random_shift_point_cloud(pc, 0.1)
        pc = pointcloud_utils.tf_jitter_point_cloud(pc, 0.01, 0.05)
        return pc, label

    def _load_data(self, data_list):
        """ load all point clouds and corresponding labels as tensor at a time """
        data = []
        label = []
        for data_file in data_list:
            package = utils.load_h5(data_file)
            data.append(package['data'][:,:self.n_points])
            label.append(package['label'][:,:self.n_points])
        data = tf.convert_to_tensor(np.concatenate(data))
        label = np.concatenate(label)
        dataset_size = label.shape[0]
        label = tf.convert_to_tensor(label)

        return data, label, dataset_size

    def label_map(self,t):
        return self._label_map[t]

    @property
    def size(self):
        return self.trainset_size, self.testset_size

    @property
    def trainset(self):
        return self._trainset

    @property
    def testset(self):
        return self._testset

if __name__=='__main__':
    batch_size = 1
    n_points = 1024
    dataset = ModelNet40H5Dataset('~/Desktop/workspace/3D/pointnet2/data/modelnet40_ply_hdf5_2048/train_files.txt',
                                  '~/Desktop/workspace/3D/pointnet2/data/modelnet40_ply_hdf5_2048/test_files.txt',
                                  batch_size, n_points)

    trainset_size, testset_size = dataset.size
    trainset = dataset.trainset

    # run one epoch
    draw_pc = pointcloud_utils.setup_pcl_viewer()
    sess = tf.Session()
    for step in range(trainset_size//batch_size):
        pc, label = sess.run([trainset.pointcloud, trainset.label])
        assert pc.shape[0]==batch_size
        print('#{}: {}'.format(step, dataset.label_map(label[0][0])))
        draw_pc(pc[0])
        import pdb
        pdb.set_trace()
