import open3d as o3d
import trimesh, os
import utils
import numpy as np
import models
import torch
import opts
import pandas as pd
#from diso import DiffMC#,DiffDMC
#from joblib import Parallel, delayed
import trimesh
import numpy as np
import concurrent.futures
import time

def load_inputs_with_bounds(shapepath, sigma, n_points):

    """
    Load input points and compute the bounding box of the pointcloud.

    Parameters
    ----------
    shapepath : str
        path to the shape data
    sigma : float
        level of noise to add to the pointcloud
    n_points : int
        number of points to sample from the pointcloud

    Returns
    -------
    shapedata : dict
        dictionary containing the pointcloud, occupancy grid, and pointcloud
    points_clean : (n_points, 3) array
        clean input points
    bounds : tuple of (3,) arrays
        bounding box of the pointcloud
    """
    shapedata = utils.load_pointcloud(shapepath)
    points_clean, _ = utils.sample_pointcloud(shapedata, N=n_points)
    noisy_points = utils.add_gaussian_noise(points_clean, sigma)
    bound_min = np.array([
        np.min(noisy_points[:, 0]), np.min(noisy_points[:, 1]),
        np.min(noisy_points[:, 2])
    ]) - 0.05
    bound_max = np.array([
        np.max(noisy_points[:, 0]), np.max(noisy_points[:, 1]),
        np.max(noisy_points[:, 2])
    ]) + 0.05
    return shapedata, points_clean, (bound_min, bound_max)


def load_occ_network( conf, ckpt, results_dir):
    occ_network = occ_network_from_conf( conf)
    occ_network.load_state_dict( torch.load (f'{results_dir}model_{ckpt}000.pth', map_location=torch.device("cuda")) )
    return occ_network
def occ_network_from_conf( conf):
    occ_network = models.NPullNetwork(**conf['model.sdf_network'])
    bias = 0.5
    occ_network.lin8 = torch.nn.Linear(in_features=256, out_features=2, bias=True)
    return occ_network
def load_state_dict(  ckpt, results_dir):
    statedict = torch.load (f'{results_dir}model_{ckpt}000.pth', map_location=torch.device("cuda"))
    return statedict

@torch.no_grad()
def uncertainty_inference (occ_network, pts):
    out = occ_network.sdf(pts.cuda()).softmax(1)
    return out[...,1]- out[...,0]

def select_ckpt(conf, args,input_points, bound_min, bound_max, ckpts):
    """
    Evaluate the given checkpoint numbers and return the one with the lowest chamfer distance wrt tot he input pointcloud.

    Parameters
    ----------
    conf : Config
        configuration object
    args : Namespace
        parsed command line arguments
    input_points : (n_points, 3) array
        input points
    bound_min : (3,) array
        lower bound of the bounding box
    bound_max : (3,) array
        upper bound of the bounding box
    ckpts : list of int
        list of checkpoint numbers to evaluate

    Returns
    -------
    best_ckpt : dict
        dictionary containing the best checkpoint number, chamfer distance, hausdorff distance, and the predicted mesh
    """
    def val_ckpt(ckpt):
        occ_network = occ_network_from_conf( conf).cuda()
        occ_network.load_state_dict( load_state_dict(  ckpt, args.results_dir) )
        sdf_function = lambda pts: -uncertainty_inference (occ_network, pts)
        cd1, hd, mesh,_= utils.validate_mesh(bound_min,bound_max, sdf_function,  resolution=conf.get_int('val.resolution'), threshold=0.0, 
                                            point_gt=input_points,
                                            N_val = conf.get_int('val.n_val'),
                                            compute_dist_fn=utils.compute_dists) 

        return {'cd1':cd1, 'hd':hd, 'mesh':mesh}
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(val_ckpt, ckpt) for ckpt in ckpts]
        print('submit took: {:.2f} sec'.format(time.time() - start))

        start = time.time()
        scores = [future.result() for future in futures]
        print('result took: {:.2f} sec'.format(time.time() - start))
    return min(scores, key=lambda x: x['cd1'])

def eval_pred_mesh(mesh, pointcloud_gt, normals_gt, n_points):
    """
    Evaluate a predicted mesh wrt a ground truth pointcloud.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        predicted mesh
    pointcloud_gt : (n_points, 3) array
        ground truth pointcloud
    normals_gt : (n_points, 3) array
        ground truth normals
    n_points : int
        number of points to sample from the mesh

    Returns
    -------
    scores : pandas.DataFrame
        a dataframe containing the chamfer-L1, chamfer-L2, and normals scores
    """
    pred_mesh_o3d = o3d.geometry.TriangleMesh( o3d.utility.Vector3dVector( mesh.vertices),
                              o3d.utility.Vector3iVector( mesh.faces) )
    pointcloud_pred = pred_mesh_o3d.sample_points_uniformly(n_points,use_triangle_normal=True)
    normals_pred = np.array(pointcloud_pred.normals).astype(np.float32)
    pointcloud_pred =np.array( pointcloud_pred.points).astype(np.float32)
    out_dict = utils.eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)
    return pd.DataFrame(out_dict, index = ["1"])[['chamfer-L1','chamfer-L2', 'normals']]

if __name__ == '__main__':  
    parser = opts.neural_pull_opts()
    parser.add_argument('--results_dir','-r', type=str, default='npull')
    parser.add_argument('--shapename', '-s',type=str, default='copyroom')
    args = parser.parse_args()
    #args.device
    os.environ['CUDA_VISIBLE_DEVICES']= str(args.device)
    conf = utils.load_conf(args.config)
    utils.fix_seeds()
    shapepath = args.shapename
    device = 'cuda'
    shapedata ,input_points, (bound_min, bound_max) = load_inputs_with_bounds(shapepath, n_points = 1024, sigma = 0.0)

    #occ_network = load_occ_network( conf, ckpt = 35, results_dir = args.results_dir).to(device)
    ckpts = range(2, 40)
    best_score = select_ckpt(conf, args, input_points, bound_min, bound_max, ckpts)

    print(eval_pred_mesh(best_score['mesh'], shapedata['pc'], shapedata['normals'], conf.get_int('val.n_val')))
    
    