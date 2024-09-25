import utils
from models import NPullNetwork
import torch
import torch.nn.functional as F
import wandb
import opts
import os
import numpy as np


def entropy(out): return -(out.softmax(1) * out.log_softmax(1)).sum(1)
def minimax_entropy (out , N_surface,lamda_min =1, lamda_max=1):
    entr = entropy(out)
    return lamda_min *entr[N_surface:].mean(0) -  lamda_max* entr[:N_surface].mean(0)
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
def spherical_init(sdf_network ):
    bias = 0.5
    sdf_network.lin8 = torch.nn.Linear(in_features=256, out_features=2, bias=True)
    torch.nn.init.normal_(sdf_network.lin8.weight[0], mean=np.sqrt(np.pi) / np.sqrt(256), std=0.0001)
    torch.nn.init.constant_(sdf_network.lin8.bias[0], -bias)
    torch.nn.init.normal_(sdf_network.lin8.weight[1], mean=-np.sqrt(np.pi) / np.sqrt(256), std=0.0001)
    torch.nn.init.constant_(sdf_network.lin8.bias[1], bias)
    return sdf_network
def main(args, conf,shapepath):
    """
    Train a neural network to predict the signed distance function of a shape.
    Args:
        args: arguments passed to the program
        conf: config file
        shapepath: path to the shape file
    Returns:
        meshes: a list of the predicted meshes at each iteration
    """
    device = 'cuda'
    utils.fix_seeds()
    # Load the shape data and create a dataset of points and normals
    shapedata, points_clean, noisy_points, (bound_min, bound_max), point, sample = utils.build_dataset(shapepath, 
                                                                                                 args.n_points,
                                                                                                 args.sigma,args.n_q)

    # Initialize the neural network
    sdf_network = NPullNetwork(**conf['model.sdf_network'])#.to(device)
    # Initialize the occupancy network
    occ_network = spherical_init(sdf_network )
    # Move the occupancy network to the specified device
    occ_network.to(device)

    # Set the learning rate
    lr = conf.get_float('train.learning_rate') if not args.fix_lr else 1e-4
    # Initialize the optimizer
    optimizer = torch.optim.Adam(occ_network.parameters(), lr=lr)
    # Initialize the scheduler
    scheduler = utils.Scheduler(optimizer = optimizer, 
                      maxiter = conf.get_int('train.maxiter'), 
                      learning_rate =lr, 
                      warm_up_end = conf.get_float('train.warm_up_end', default=0.0))

    # Set the ground truth points
    # Set the iteration step
    iter_step , eps  = 0, 1e-12
    # Set the number of iterations
    res_step = conf.get_int('train.maxiter') - iter_step

    # Set the lambda values for the minimax loss
    lamda_min, lamda_max = (1,10 )if args.lamda_max is None else (1, args.lamda_max)
    # Set the query samples
    query_samples = torch.empty((args.n_surface  + args.n_queries , 3), device = 'cuda')
    
    # Set the learning rate scheduler
    beta = 8*np.log(10) /args.n_minimax
    infoloss_scheduler = np.exp(- beta*np.arange(res_step) ) 

    # Train the network
    for iter_i in range(iter_step, res_step):
        # Update the learning rate
        if not args.fix_lr:
            scheduler.update_learning_rate_np(iter_i)
        # Sample query points to pull
        loss = 0
        loss_sdf = torch.zeros(1)
        # Sample points and samples
        points, samples,_ = utils.np_train_data(point, sample, conf.get_int('train.batch_size'))
        
        # Compute the gradients of the uncertainty function at the sample points
        samples.requires_grad = True
        out_sample = occ_network.sdf(samples).softmax(1)
        sdf_sample = -(out_sample[:,1]-out_sample[:,0])
        gradients_sample = utils.gradient(samples, sdf_sample).squeeze()
        # Compute the gradient norm
        grad_denom = gradients_sample.norm(2, 1, keepdim=True).clamp_min(eps).expand_as(gradients_sample)
        grad_norm = gradients_sample /grad_denom # F.normalize(gradients_sample, dim=1)
        # If the stop grad flag is set, detach the gradient norm
        if args.stop_grad:
            grad_norm = grad_norm.detach()
        # Pull the sample points towards the surface
        sample_moved = samples - grad_norm * sdf_sample.unsqueeze(-1)                 # 5000x3
        # Compute the loss
        loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean()
        loss = 10*loss_sdf
        # Sample additional query points
        queries =  utils.sample_uniform_points(boxsize = max(bound_max)-min(bound_min) , n_points_uniform = args.n_queries)
        # Set the query samples
        query_samples[:args.n_surface] = torch.from_numpy(subsample_pointcloud(noisy_points, args.n_surface) ).float().cuda()
        query_samples[args.n_surface:] = queries
        # Compute the SDF values at the query samples
        sdf_queries = occ_network.sdf(query_samples)  
        # Compute the minimax loss
        info_loss  = minimax_entropy (sdf_queries , args.n_surface, lamda_min, lamda_max)
        # Compute the total loss
        loss = infoloss_scheduler [iter_i] *info_loss+loss
        
        # Zero the gradients
        scheduler.optimizer.zero_grad()
        # Backpropagate the loss
        loss.backward()
        # Update the network parameters
        scheduler.optimizer.step()

        # Increment the iteration step
        iter_step += 1
        # Log the loss
        if args.wandb_log:
            wandb.log({'iter_i': iter_i, 'loss': loss_sdf.item()})
        # Save the model
        if iter_step % conf.get_int('train.save_freq') == 0 and iter_step!=0: 
            # Save the model
            state_dict =  {k: v.cpu() for k, v in occ_network.state_dict().copy().items()}
            torch.save(state_dict,f'{args.exp_dir}/model_{iter_step}.pth' )
            print(f'save model at {args.exp_dir}/model_{iter_step}.pth')

if __name__ == '__main__':  
    args = opts.neural_pull_opts().parse_args()
    #args.device
    os.environ['CUDA_VISIBLE_DEVICES']= str(args.device)
    conf = utils.load_conf(args.config)
    if args.wandb_log:
        utils.init_wandb (name = args.name, config = conf)
    os.makedirs(args.exp_dir, exist_ok=True)
    main(args, conf,args.shapepath)
