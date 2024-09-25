# Unsupervised Occupancy Learning from Sparse Point Cloud


This repository contains the implementation of the CVPR 2024 paper [Unsupervised Occupancy Learning from Sparse Point Cloud](https://arxiv.org/pdf/2404.02759)  by Amine Ouasfi and Adnane Boukhayma.

![](./poster.jpeg)
## Overview
------------

This paper proposes a novel approach for unsupervised occupancy learning from sparse point clouds. The method learns to predict the occupancy of a 3D space from a sparse set of points, without requiring any supervision or prior knowledge of the scene. 

## Repository Structure
------------------------

The repository is organized as follows:

* `models`: contains the implementation of the neural network architecture used in the paper.
* `utils`: contains various utility functions used throughout the code, including data loading, processing, and visualization.
* `train_socc.py`: contains the training script for the neural network.
* `eval.py`: contains the evaluation script for the neural network.
* `README.md`: this file.

## Dependencies
------------

The code is written in Python and requires the following dependencies:

* `torch`: the PyTorch library for deep learning.
* `numpy`: the NumPy library for numerical computations.
* `scipy`: the SciPy library for scientific computing.
* `trimesh`: the Trimesh library for 3D mesh processing.
* `open3d`: the Open3D library for 3D point cloud processing.
* `wandb`: the Weights & Biases library for experiment tracking.

## Data
-----

### ShapeNet
We use a subset of the [ShapeNet](https://shapenet.org/) data as chosen by [Neural Splines](https://github.com/fwilliams/neural-splines). This data is first preprocessed to be watertight as per the pipeline in the [Occupancy Networks repository](https://github.com/autonomousvision/occupancy_networks), who provide both the pipleline and the entire preprocessed dataset (73.4GB). 

The Neural Spline split uses the first 20 shapes from the test set of 13 shape classes from ShapeNet.You can download the dataset (73.4 GB) by running the [script](https://github.com/autonomousvision/occupancy_networks#preprocessed-data) from Occupancy Networks. After, you should have the dataset in `data/ShapeNet` folder.


### Faust

The Faust Dataset can be downloaded from the [official website ](https://faust-leaderboard.is.tuebingen.mpg.de). We followed the preprocessing steps outlined in [Occupancy Networks repository](https://github.com/autonomousvision/occupancy_networks). Specifically, we normalized the meshes to the unit cube and uniformly sampled 100,000 points with their corresponding normals for evaluation.

### Surface Reconstruction Benchamark data
The Surface Reconstruction Benchmark (SRB) data is provided in the [Deep Geometric Prior repository](https://github.com/fwilliams/deep-geometric-prior).

If you use this data in your research, make sure to cite the Deep Geometric Prior paper.

## Training
------------

To train the neural network, run the following command:
```bash
python train_socc.py --device <device>  --n_points <n_points> --sigma <sigma> --name <experiment_name> --n_surface <n_surface> --lamda_max <lamda_max> --n_queries <n_queries> --n_minimax <n_minimax> --shapepath <shapepath> --exp_dir <experiment_directory>
```
This will train the network using the configuration specified in `config.json` and store the trained model in the `results` directory.

For ShapeNet we used: 
```bash
python train_socc.py  --device <device>  --shapepath <shapepath> --exp_dir <experiment_directory> -name <experiment_name> --n_points 1024 --sigma 0.005  --n_surface 1000 --lamda_max 10 --n_minimax 10000
```
For Faust example provided in `data` we used: 
```bash
python train_socc.py --shapepath data/Faust/real_11/   --device 0 --exp_dir experiments/Faust/real_11/   --n_points 1024 --sigma 0.0  --n_surface 1000 --lamda_max 10  --n_minimax 1000 
```
## Evaluation
-------------

To evaluate the trained model, run the following command:
```bash
python eval.py  --device 0 --shapename <shapename> --results_dir results/
```
This will evaluate the model on the test set and store the results in the `results` directory.

## Configuration
-------------

The configuration file `configs/conf.conf` contains the following parameters:

* `num_points`: the number of points to sample from the point cloud.
* `num_queries`: the number of queries to generate for each point.
* `sigma`: the standard deviation of the noise added to the point cloud.
* `batch_size`: the batch size used for training.
* `num_epochs`: the number of epochs to train for.
* `learning_rate`: the learning rate used for training.
* `device`: the device to use for training (e.g. `cuda:0`).

## Citation
------------

If you use this code in your research, please cite the following paper:
```
@inproceedings{SparseOcc,
  author = {Amine Ouasfi and Adnane Boukhayma},
  title = {Unsupervised Occupancy Learning from Sparse Point Cloud},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
}
```


