# pointnet.tensorflow
This is just a neater reimplementation of [PointNet](http://stanford.edu/~rqi/pointnet/) and [PointNet++](http://stanford.edu/~rqi/pointnet2/) using tensorflow on ModelNet40.

### Prerequisite
- please check [pointnet](https://github.com/charlesq34/pointnet) and [pointnet2](https://github.com/charlesq34/pointnet2). It should be similar
- [optional] I have used [vispy](http://vispy.org/) for a simple point cloud visualization

### How to run
- download dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and put the uncompressed data into directory ```data```
- remember to change ```$CUDA_ROOT``` and ```$TF_ROOT``` in the compile script if needed
- if you are using virtual environment or conda, remember to compile the tf ops while the environment activated
```
# compile tensorflow ops
$ cd ${POINTNET-TENSORFLOW-HOME}/utils/tf_sampling
$ sh tf_sampling_compile.sh 
$ cd ${POINTNET-TENSORFLOW-HOME}/utils/tf_grouping
$ sh tf_grouping_compile.sh
# run training
$ cd ${POINTNET-TENSORFLOW-HOME}
$ python train.py --work_dir ${dir-path-to-save-log-and-checkpoint}
```

### Notes
- I mainly change the dataloader and it took about 2.4 hours for training PoinNet and 3.7 hours for training PointNet++ on NVIDIA 1080TI
- I am using tensorflow 1.8 but it should go well on other version
- Disclaimer: this repo is just a quick implementation to let me get deeper in PointNet and refresh my tensorflow coding. Thus I just did a pretty incomplete test on it (similar classification accuracy claimed in the paper)

### Credit
This repo largely depends on [pointnet](https://github.com/charlesq34/pointnet) and [pointnet2](https://github.com/charlesq34/pointnet2), and so does the dataset.
