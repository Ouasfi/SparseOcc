import argparse

def neural_pull_opts(nap = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/conf.conf')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--shapepath', type=str, default='path to a single shape or a split file') ## path to a single shape or a split file
    parser.add_argument('--exp_dir', type=str, default='experiments/') ## path to a single shape or a split file
    parser.add_argument('--n_points', type=int, default=3000)
    parser.add_argument('--n_q', type=float, default=1.)
    parser.add_argument('--sigma', type=float, default=0.00)
    parser.add_argument("--wandb_log", help="log with wandb",
                    action="store_true")
    parser.add_argument("--stop_grad", help="detach the gradient in the Newton Iteration",
                    action="store_true")
    parser.add_argument("--fix_lr", help="do not schedule lr ",action="store_true")
    parser.add_argument("--grad_const", help="do not schedule lr ",action="store_true")
#    parser.add_argument("--save", help="save model, mesh ",action="store_true")
    parser.add_argument("--siren", help="siren model ",action="store_true")
    ## trimming term 
    parser.add_argument("--trimloss", help="trimming loss ",action="store_true")
    ## entropy
    parser.add_argument('--n_surface', type=int, default=1000)
    parser.add_argument('--n_minimax', type=int, default=10000)

    parser.add_argument('--n_queries', type=int, default=10000)
    parser.add_argument('--lamda_max', type=float, default=1)
    
    
   
    return parser




