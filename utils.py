import numpy as np
import torch
import glob 
import trimesh
import torch.nn.functional as F
import math
import random
import secrets
from scipy.spatial import cKDTree
import mcubes
from pyhocon import ConfigFactory
import wandb


def fix_seeds():
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
def load_conf(path):
    """
    params:
    ------
    path: path to the neural pull config file
    
    returns the config namespace
    """
    f = open(path)
    conf_text = f.read()
    f.close()

    return ConfigFactory.parse_string(conf_text)

def load_pointcloud(datapath):
    try: 
        dataspace = np.load(datapath + 'points.npz')
        points_tgt = dataspace['points'].astype(np.float32)
        occ_tgt = np.unpackbits(dataspace['occupancies']).astype(np.float32)
    except : 
        points_tgt, occ_tgt = None, None
    datashape = np.load(datapath + 'pointcloud.npz')
    pointcloud_tgt = datashape['points'].astype(np.float32)
    

    
    normals_tgt = datashape.get('normals', np.zeros_like(pointcloud_tgt)).astype(np.float32) 
    data = {
        'points': points_tgt,
        'occ' : occ_tgt,
        'pc': pointcloud_tgt,
        'normals': normals_tgt,
           }
    data['bounds']= datashape.get('bounds')
    return data

def sample_pointcloud(data, N):
    """
    params:
    ------
    data: dict containing points and normals.
    N : int number of points to sample.
    
    returns sampled points and normals
    """
    scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
    rng = np.random.default_rng( scr )     
    pointcloud = data['pc']
    normals_tgt = data['normals']
    point_idx = rng.choice(pointcloud.shape[0], N, replace = False)
    return pointcloud[point_idx,:], normals_tgt[point_idx,:]

def sample_pointcloud_srb(data, N):
    """
    params:
    ------
    data: dict containing points and normals.
    N : int number of points to sample.
    
    returns sampled points and normals
    """
    scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
    rng = np.random.default_rng( scr )     
    pointcloud = data['pc']
    normals_tgt = data['normals']
    point_idx = rng.choice(pointcloud.shape[0], N, replace = False)
    points, normals = pointcloud[point_idx,:], normals_tgt[point_idx,:]
    cp = points.mean(axis=0)
    points = points - cp[None, :]
    # scale = np.linalg.norm(points, axis=-1).max(-1)
    scale = np.abs(points).max()
    points = points / scale

    return points, normals, (cp, scale)
def add_gaussian_noise(points, sigma ):
    """
    params:
    ------
    points: clean input points of size( N, 3)
    sigma: std.
    
    returns noisy input points.
    """
    return points + sigma* np.random.randn(points.shape[0],points.shape[-1])

def sample_shape(path, classe):
    """
    params:
    ------
    path : path to the dataset directory 
    classe : classe to sample from
    
    return path to the sampled shape
    """
    scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
    rng = np.random.default_rng( scr ) 
    return rng.choice(glob.glob(path + classe + '/*/'))

def np_train_data(point, sample, batch_size, device = 'cuda'):
    index_coarse = np.random.choice(10, 1)
    index_fine = np.random.choice((sample.shape[0]-1)//10 , batch_size, replace = False)
    index = index_fine * 10 + index_coarse
    points = point[index]#.unsqueeze(0)
    samples = sample[index]#.unsqueeze(0)
    return points.to(device), samples.to(device), index

def get_sigmas(noisy_data):
    
    sigma_set = []

    ptree = cKDTree(noisy_data)

    for p in np.array_split(noisy_data, 100, axis=0):
        d = ptree.query(p, 50 + 1)
        sigma_set.append(d[0][:, -1])

    sigmas = np.concatenate(sigma_set)
    local_sigma = torch.from_numpy(sigmas).float().cuda()
    return local_sigma
def fast_process_data(pointcloud, n_queries = 1):
    """
    params:
    ------
    pointcloud: input pointcloud.
    
    returns a dict containing query points sampled around the input and their corresponding nearst neighbors.
    """
    dim = pointcloud.shape[-1]
    scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
    rng = np.random.default_rng( scr ) 
    pointcloud_ = pointcloud 
    POINT_NUM, POINT_NUM_GT,  = pointcloud.shape[0] // 60 , pointcloud.shape[0] // 60 * 60 
    QUERY_EACH = int(n_queries*1000000//POINT_NUM_GT)
    print(POINT_NUM,POINT_NUM_GT,QUERY_EACH)
    scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
    # Subsample to n_points_gt
    point_idx = rng.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
    pointcloud = pointcloud_[point_idx,:]
    ptree = cKDTree(pointcloud)
    sigmas = ptree.query(pointcloud,51,n_jobs=10)[0][:,-1]
    ## Compute NN per input 
    sample = pointcloud.reshape(1,POINT_NUM_GT,dim) + scale*np.expand_dims(sigmas,-1) * rng.normal(0.0, 1.0, size=(QUERY_EACH, POINT_NUM_GT, dim))
    n_idx = ptree.query(sample.reshape(-1,dim),1,n_jobs=10)[1]
    sample_near =  pointcloud[n_idx].reshape((QUERY_EACH, POINT_NUM_GT, dim))
    return { "sample": sample, 'point' : pointcloud,'gt_point' : pointcloud_, 
            'sample_near' : sample_near, 'idx': point_idx, 'rho_idx': n_idx}


import napf
def compute_dists_flann(recon_points, gt_points):
    recon_kd_tree = napf.KDT(tree_data=recon_points, metric=2) 
    gt_kd_tree = napf.KDT(tree_data=gt_points, metric=2)
    re2gt_distances, indices = recon_kd_tree.knn_search(
                            queries=gt_points,
                            kneighbors=1,
                            nthread=50)
    gt2re_distances, indices = gt_kd_tree.knn_search(
                            queries=recon_points,
                            kneighbors=1,
                            nthread=50)
    
    re2gt_distances = np.sqrt(re2gt_distances)
    gt2re_distances = np.sqrt(gt2re_distances)
    cd_re2gt = np.mean(re2gt_distances)
    cd_gt2re = np.mean(gt2re_distances)
    hd_re2gt = np.max(re2gt_distances)
    hd_gt2re = np.max(gt2re_distances)
    chamfer_dist = 0.5* (cd_re2gt + cd_gt2re)
    hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    return chamfer_dist , hausdorff_distance

def validate_mesh(bound_min,bound_max,query_func,  resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None,N_val = 100000, compute_dist_fn=compute_dists_flann ):
    #N_val = 100000

    bound_min = torch.tensor(bound_min, dtype=torch.float32)
    bound_max = torch.tensor(bound_max, dtype=torch.float32)
    mesh = extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, 
                            query_func=query_func)
    recon_points = mesh.sample(N_val)
    cd1, hd = compute_dist_fn(point_gt, recon_points)
    return cd1, hd, mesh,recon_points
def extract_fields( bound_min, bound_max, resolution, query_func):
    N = 32
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry( bound_min, bound_max, resolution, threshold, query_func):
    #print('Creating mesh with threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    mesh = trimesh.Trimesh(vertices, triangles)

    return mesh

from scipy.spatial import cKDTree as KDTree
def compute_dists(recon_points, gt_points):
    recon_kd_tree = KDTree(recon_points)
    gt_kd_tree = KDTree(gt_points)
    re2gt_distances, re2gt_vertex_ids = recon_kd_tree.query(gt_points, n_jobs=10)
    gt2re_distances, gt2re_vertex_ids = gt_kd_tree.query(recon_points, n_jobs=10)

    cd_re2gt = np.mean(re2gt_distances)
    cd_gt2re = np.mean(gt2re_distances)
    hd_re2gt = np.max(re2gt_distances)
    hd_gt2re = np.max(gt2re_distances)
    chamfer_dist = 0.5* (cd_re2gt + cd_gt2re)
    hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    return chamfer_dist , hausdorff_distance

def build_dataset_srb(shapepath:str, n_points:int, sigma:float, n_queries = 1 ):
    """
    sample the input pointcloud and the supervision points
    params:
    ------
    shapepath: path to the pointcloud npz file
    n_points: size of the input pointcloud
    sigma: level of noise to apply to the pointcloud    
    """
    shapedata = load_pointcloud(shapepath)
    points_clean, normals, (cp, scale) = sample_pointcloud_srb(shapedata, N = n_points)
    noisy_points = add_gaussian_noise(points_clean, sigma )
    shapedata['cp'], shapedata['scale'] = (cp, scale)
    datanp = fast_process_data(noisy_points,n_queries)
    np_point = np.asarray(datanp['sample_near']).reshape(-1,3)
    point = torch.from_numpy( np.asarray(datanp['sample_near']).reshape(-1,3) ).to(torch.float32)#.to(device)
    sample = torch.from_numpy( np.asarray(datanp['sample']).reshape(-1,3) ).to(torch.float32)#.to(device)
    bound_min = np.array([np.min(np_point[:,0]), np.min(np_point[:,1]), np.min(np_point[:,2])]) -0.05
    bound_max = np.array([np.max(np_point[:,0]), np.max(np_point[:,1]), np.max(np_point[:,2])]) +0.05
    return shapedata, datanp, noisy_points, (bound_min, bound_max), point, sample


def build_dataset(shapepath:str, n_points:int, sigma:float, n_queries = 1 ):
    """
    sample the input pointcloud and the supervision points
    params:
    ------
    shapepath: path to the pointcloud npz file
    n_points: size of the input pointcloud
    sigma: level of noise to apply to the pointcloud    
    """
    shapedata = load_pointcloud(shapepath)
    points_clean, normals = sample_pointcloud(shapedata, N = n_points)
    noisy_points = add_gaussian_noise(points_clean, sigma )

    datanp = fast_process_data(noisy_points,n_queries)
    np_point = np.asarray(datanp['sample_near']).reshape(-1,3)
    point = torch.from_numpy( np.asarray(datanp['sample_near']).reshape(-1,3) ).to(torch.float32)#.to(device)
    sample = torch.from_numpy( np.asarray(datanp['sample']).reshape(-1,3) ).to(torch.float32)#.to(device)
    bound_min = np.array([np.min(np_point[:,0]), np.min(np_point[:,1]), np.min(np_point[:,2])]) -0.05
    bound_max = np.array([np.max(np_point[:,0]), np.max(np_point[:,1]), np.max(np_point[:,2])]) +0.05
    return shapedata, datanp, noisy_points, (bound_min, bound_max), point, sample

def init_wandb( name = "Baseline", config= {}):
    wandb.init(
        # set the wandb project where this run will be logged
        project="neural_pull",
        name = name,
        # track hyperparameters and run metadata
        config=config
    )
def sample_uniform_points(boxsize = 1.01, n_points_uniform = 5000):
    points_padding = 0.01

    #boxsize = 1 + points_padding
    points_uniform = torch.rand(n_points_uniform, 3, device = 'cuda')
    points_uniform = boxsize * (points_uniform - 0.5)
    return points_uniform
def pull_points (sdf_network, samples, alpha = 1.):
    """
    pull points towards the surface using the sdf of the network and it's gradient
    """
    gradients_sample = sdf_network.gradient(samples).squeeze() # 5000x3
    sdf_sample = sdf_network.sdf(samples)                      # 5000x1
    grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
    sample_moved = samples -alpha* grad_norm * sdf_sample
    return sample_moved


import torch
from torch.autograd import grad
def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad
def get_exp(args):
    return f'{args.method}_p_{args.n_points}_sigma_{args.sigma}_rho_{args.rho}'

import sys, os
sys.path.append(os.path.abspath("../convolutional_occupancy_networks") )
from src.eval import MeshEvaluator
import pandas as pd
evaluator = MeshEvaluator(n_points=100000)
def eval_mesh(mesh,shapedata ):
    """
    computes evaluation metrics using the predicted mesh and uniform query points with gt occupancies.
    params:
    ------
    mesh: predicted trimesh mesh
    shapedata: dict containing gt normals,input pointcloud,  query points and their occupancy values
    """
    eval_dict_mesh = evaluator.eval_mesh(
                        mesh, shapedata['pc'], shapedata['normals'], shapedata['points'], 
                        shapedata['occ'], remove_wall=False)
    return pd.DataFrame(eval_dict_mesh, index = {0})

import torch
import torch.nn as nn




def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = recon_kd_tree = napf.KDT(tree_data=points_tgt, metric=2)  
    dist, idx =   recon_kd_tree.knn_search(
                            queries=points_src,
                            kneighbors=1,
                            nthread=50) 
    idx = np.squeeze(idx)
    dist = np.sqrt(dist)
    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}
def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None,
                        thresholds=np.linspace(1./1000, 1, 1000)):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            #logger.warn('Empty pointcloud / mesh detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        #recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        #precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        #F = [
         #   2 * precision[i] * recall[i] / (precision[i] + recall[i])
         #   for i in range(len(precision))
        #]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            #'f-score': F[9], # threshold = 1.0%
            #'f-score-15': F[14], # threshold = 1.5%
            #'f-score-20': F[19], # threshold = 2.0%
        }

        return out_dict


class Scheduler:
    def __init__(self, optimizer, maxiter, learning_rate, warm_up_end):
        self.warm_up_end = warm_up_end
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.optimizer = optimizer
    def get_lr(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        return lr
    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr