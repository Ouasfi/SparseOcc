import utils
from models import NPullNetwork
import torch
import torch.nn.functional as F
import wandb
import opts
import os
import numpy as np

@torch.no_grad()
def sdf_inference(sdf_network, pts):
    out = sdf_network.sdf(pts.cuda()).softmax(1)
    return out[...,1]- out[...,0]
def entropy(out): return -(out.softmax(1) * out.log_softmax(1)).sum(1)
def minimax_entropy (out , N_surface,lamda_min =1, lamda_max=1):
    entr = entropy(out)
    return lamda_min *entr[N_surface:].mean(0) -  lamda_max* entr[:N_surface].mean(0)
def sample_uniform_points(boxsize = 1.01, n_points_uniform = 5000):
    points_padding = 0.01

    #boxsize = 1 + points_padding
    points_uniform = torch.rand(n_points_uniform, 3, device = 'cuda')
    points_uniform = boxsize * (points_uniform - 0.5)
    return points_uniform
def subsample_pointcloud(pointcloud, N):
    """
    params:
    ------
    data: dict containing points and normals.
    N : int number of points to sample.
    
    returns sampled points and normals
    """
    scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
    rng = np.random.default_rng( scr )     
    
    point_idx = rng.choice(pointcloud.shape[0], N, replace = False)
    return pointcloud[point_idx,:]
def main(args, conf,shapepath):
    device = 'cuda'
    utils.fix_seeds()
    shapedata, points_clean, noisy_points, (bound_min, bound_max), point, sample = utils.build_dataset(shapepath, 
                                                                                                 args.n_points,
                                                                                                 args.sigma,args.n_q)
    print((bound_min, bound_max))
    sdf_network = NPullNetwork(**conf['model.sdf_network'])#.to(device)
    lr = conf.get_float('train.learning_rate') if not args.fix_lr else 1e-4
    bias = 0.5
    sdf_network.lin8 = torch.nn.Linear(in_features=256, out_features=2, bias=True)
    torch.nn.init.normal_(sdf_network.lin8.weight[0], mean=np.sqrt(np.pi) / np.sqrt(256), std=0.0001)
    torch.nn.init.constant_(sdf_network.lin8.bias[0], -bias)
    torch.nn.init.normal_(sdf_network.lin8.weight[1], mean=-np.sqrt(np.pi) / np.sqrt(256), std=0.0001)
    torch.nn.init.constant_(sdf_network.lin8.bias[1], bias)
    sdf_network.to(device)
    optimizer = torch.optim.Adam(sdf_network.parameters(), lr=lr)
    scheduler = utils.Scheduler(optimizer = optimizer, 
                      maxiter = conf.get_int('train.maxiter'), 
                      learning_rate =lr, 
                      warm_up_end = conf.get_float('train.warm_up_end', default=0.0))
    gt_points = shapedata['pc']
    iter_step = 0
    eps = 1e-12
    res_step = conf.get_int('train.maxiter') - iter_step
    mcubes_threshold = 0.0
    meshes, iters, scores, scores_s = [], [], [], []
    state_dicts = []
    N_surface = 10000  if args.n_surface is None else args.n_surface #N_points
    batch_queries = 1000 if args.n_queries is None else args.n_queries
    lamda_min, lamda_max = (1,10 )if args.lamda_max is None else (1, args.lamda_max)
    query_samples = torch.empty((N_surface +batch_queries , 3), device = 'cuda')
    margin_pulling = True
    minimax_warmup = True
    n_minimax = 10000 if args.n_minimax is None else args.n_minimax
    beta = 8*np.log(10) /n_minimax
    infoloss_scheduler = np.exp(- beta*np.arange(res_step) ) 
    for iter_i in range(iter_step, res_step):
        if not args.fix_lr:
        ## update the learning rate
            scheduler.update_learning_rate_np(iter_i)
        ## sample query points to pull
        loss = 0
        loss_sdf = torch.zeros(1)
        points, samples,_ = utils.np_train_data(point, sample, conf.get_int('train.batch_size'))
        
        samples.requires_grad = True
        out_sample = sdf_network.sdf(samples).softmax(1)
        sdf_sample = -(out_sample[:,1]-out_sample[:,0])
        gradients_sample = utils.gradient(samples, sdf_sample).squeeze()
        grad_denom = gradients_sample.norm(2, 1, keepdim=True).clamp_min(eps).expand_as(gradients_sample)#.pow(2)
        grad_norm = gradients_sample /grad_denom # F.normalize(gradients_sample, dim=1)
        if args.stop_grad:
            grad_norm = grad_norm.detach()
        sample_moved = samples - grad_norm * sdf_sample.unsqueeze(-1)                 # 5000x3
        loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean()
        loss = 10*loss_sdf
        ####
        queries =  sample_uniform_points(boxsize = max(bound_max)-min(bound_min) , 
                                            n_points_uniform = batch_queries)
        query_samples[:N_surface] = torch.from_numpy(subsample_pointcloud(noisy_points, N_surface) ).float().cuda()
        query_samples[N_surface:] = queries
        sdf_queries = sdf_network.sdf(query_samples)  
        info_loss  = minimax_entropy (sdf_queries , N_surface, lamda_min, lamda_max)
        loss = infoloss_scheduler [iter_i] *info_loss+loss
        
        scheduler.optimizer.zero_grad()
        loss.backward()
        scheduler.optimizer.step()

        iter_step += 1
        if args.wandb_log:
            wandb.log({'iter_i': iter_i, 'loss': loss_sdf.item()})
        if iter_step % conf.get_int('train.val_freq') == 0 and iter_step!=0: 
            try:
                if args.adaptive_thresh:
                    mcubes_threshold = sdf_network(torch.from_numpy(noisy_points).to(device).float()).abs().median().item()
                cd1, hd, mesh,recon_points = utils.validate_mesh(bound_min,bound_max, lambda pts: sdf_inference(sdf_network, pts)
                ,resolution=256, threshold=mcubes_threshold, point_gt=gt_points, iter_step=iter_step, logger=None)
                #recon_points = mesh.sample(100000)
                cd1_s, hd_s = utils.compute_dists(noisy_points, recon_points)
                meshes.append(mesh)
            except:
                cd1, hd = 0.1,0.1 #default value
                cd1_s, hd_s = 0.1,0.1 #default value
                meshes.append(None)

            iters.append(iter_step)
            scores.append(cd1)
            scores_s.append(cd1_s)
            state_dicts.append( {k: v.cpu() for k, v in sdf_network.state_dict().copy().items()})
            if args.save:
                mesh.export(f'{args.exp_dir}/mesh_{iter_step}.off')
                torch.save(state_dicts[-1],f'{args.exp_dir}/model_{iter_step}.pth' )
            if args.wandb_log:
                wandb.log({'Chamfer L1': cd1, 'Hausdorf': hd})
                wandb.log({'Chamfer L1_s': cd1_s, 'Hausdorf_s': hd_s,})
    return meshes,iters,scores,scores_s,shapedata, state_dicts
if __name__ == '__main__':  
    args = opts.neural_pull_opts().parse_args()
    #args.device
    os.environ['CUDA_VISIBLE_DEVICES']= str(args.device)
    conf = utils.load_conf(args.config)
    if args.wandb_log:
        utils.init_wandb (name = args.name, config = conf)
    os.makedirs(args.exp_dir, exist_ok=True)
    main(args, conf,args.shapepath)
