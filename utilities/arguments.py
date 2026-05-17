import argparse


def args_setting():
    parser = argparse.ArgumentParser(description='Args')


    parser.add_argument('--seed', type=int, default=0)


    parser.add_argument('--task', default="hopper-medium-v2",
                        help='D4RL environment (default: hopper-medium-v2)')


    parser.add_argument('--dynamics_model_name', default='BatchLinear', 
                        help='the name of environment dynamics model')
    parser.add_argument('--dynamics_path', default='./weight/',
                        help='path to load saved dynamics model (default: None)')
    parser.add_argument('--dynamics_save_path', default='./weight/',
                        help='path to save dynamics model (default: None)')

    parser.add_argument('--ensemble_size', type=int, default=100,
                        help='size of dynamics ensemble (default: 100)')
    parser.add_argument('--transition_layer_size', type=int, default=512,
                        help='hidden size per layer of dynamics model (default: 256)')
    parser.add_argument('--transition_layers', type=int, default=4,
                        help='number of hidden layers of dynamics model (default: 4)')

    parser.add_argument('--transition_num_epoch', type=int, default=1000,
                        help='number of epochs to train dynamics model (default: 1000)')
    parser.add_argument('--transition_batch_size', type=int, default=512,
                        help='batch size of dynamics loss (default: 256)')
    parser.add_argument('--transition_lr', type=float, default=1e-4,
                        help='dynamics learning rate (default: 1e-4)')


    parser.add_argument('--policy_type', default="Gaussian",
                        help='policy type: Gaussian')
    parser.add_argument('--det_policy', action="store_true",
                        help='deterministic policy, valid for policy_type=Gaussian (default: False)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau_value', type=float, default=5e-3,
                        help='smoothing coefficient for value target (default: 5e-3)')
    parser.add_argument('--tau_policy', type=float, default=1e-5,
                        help='smoothing coefficient for policy target (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='weight alpha to determine the relative importance of the KL\
                              term against the reward (default: 0.1)')
    parser.add_argument('--automatic_alpha_tuning', action="store_true", default=True,
                        help='automaically adjust alpha (default: True)')
    parser.add_argument('--target_kld', type=float, default=5.,
                        help='target KL divergence when automatically adjusting alpha (default: 5.)')

    parser.add_argument('--agent_num_steps', type=int, default=int(2e6),
                        help='number of gradient steps for policy learning (default: 2e6)')
    parser.add_argument('--real_batch_size', type=int, default=256,
                        help='batch size of MDP loss (default: 128)')
    parser.add_argument('--adv_batch_size', type=int, default=256,
                        help='batch size of AMG loss (default: 128)')

    parser.add_argument('--actor_lr', type=float, default=3e-5,
                        help='actor learning rate (default: 3e-5)')
    parser.add_argument('--critic_lr', type=float, default=3e-4,
                        help='critic learning rate (default: 3e-4)')
    parser.add_argument('--n_step', type=int, default=64,
                        help="N-step TD. Active only in Antmaze")
    
    parser.add_argument('--use_bc_regularization', action='store_true', default=False)
    parser.add_argument('--bc_weight', type=float, default=0.1, 
                        help='bc loss coefficient')

    parser.add_argument('--agent_layer_size', type=int, default=256,
                        help='hidden size of actor and critic (default: 256)')

    parser.add_argument('--explore_ratio', type=float, default=0.,
                        help='exploration probability of primary player (default:0.)')


    parser.add_argument('--use_adaptive_belief', action="store_true", default=False, 
                        help="whether to use belief state")
    parser.add_argument('--ablation_belief', action='store_true', default=False,
                        help="Ablation belief effect")
    parser.add_argument('--adv_horizon', type=int, default=1000,
                        help='horizon (default: 1000)')
    parser.add_argument('--adv_explore_ratio', type=float, default=0.1,
                        help='exploration probability of adversarial player (default: 0.1)')
    parser.add_argument('--num_sample_transition', type=int, default=10,
                        help='hyperparameter N for dynamics sampling (default: 10)')
    parser.add_argument('--order_transition', type=int, default=2,
                        help='hyperparameter k for dynamics sampling (default: 2)')
    parser.add_argument('--dynamics_reweight', action='store_true', default=True)
    parser.add_argument('--reweight_explore_ratio', type=float, default=0.1)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--MC_size_state', type=int, default=10,
                        help='state sample size for estimating expectation (default: 10)')
    parser.add_argument('--MC_size_action', type=int, default=20,
                        help='action sample size for estimating expectation (default: 20)')
    parser.add_argument('--real_ratio', type=float, default=0.5, 
                        help="The weight of real sample during updating")

    parser.add_argument('--lamba', type=float, default=0.9,
                        help="balance the finsher matrix and natural gradient matrix")
    parser.add_argument('--bregman_reg', action="store_true", default=False)
    parser.add_argument('--beta', type=float, default=0.999,
                        help="the decay rate of matrix F_k used in Bregman divergence")
    parser.add_argument('--L2_max_grad', type=float, default=0.5,
                        help="maximum gradient used in bregman regularization")
    parser.add_argument('--proj_lr', type=float, default=7.5e-4,
                        help="direction of projection")
    parser.add_argument('--grad_factor', type=float, default=0.6,
                        help='scaling the actor lr into bregman lr')

    parser.add_argument('--cpu', action="store_true", default=False, 
                        help='run on CPU (default: False)')
    parser.add_argument('--optim', default="Adam",
                        help="Optimizer of policy and critic")
    parser.add_argument('--eval', action="store_false", default=True,
                        help='periodically evaluate on real environment (default: True)')
    parser.add_argument("--mode", default= 'local', 
                        help="prediction mode")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="The loop steps of real env during evaluation (antmaze:1w,otherwise:1k)")
    parser.add_argument("--log_loss_path", default='./log/dynamics_loss/')
    parser.add_argument("--log_policy_info", default='./log/policy_info/')
    parser.add_argument("--tuning_info", default="", help="hyperparameter tuning information")
    parser.add_argument("--record_time", action="store_true", default=False, help="whether to record time")
    parser.add_argument('--log_belief', action="store_true", default=False, help="whether to record belief and suprise")



    return parser.parse_args()

def print_args(args, file_path=None):
    if file_path is None:
        print('------------------------ arguments ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))

        for arg in str_list:
            print(arg, flush=True)
        print('-------------------- end of arguments ---------------------', flush=True)
    else:
        with open(file_path, 'a') as f:
            print('------------------------ arguments ------------------------', flush=True, file=f)
            str_list = []
            for arg in vars(args):
                dots = '.' * (48 - len(arg))
                str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))

            for arg in str_list:
                print(arg, flush=True, file=f)
            print('-------------------- end of arguments ---------------------', flush=True, file=f)
