from utils.optimizers import Fromage, RAdam, AdamW, AdaBelief, Yogi, MSVAG
from torch import optim

def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.opt_momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                          weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.optim == 'fromage':
        return Fromage(model_params, args.lr)
    elif args.optim == 'radam':
        return RAdam(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                          weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.optim == 'adamw':
        return AdamW(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                          weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.optim == 'adabelief':
        return AdaBelief(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                          weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.optim == 'yogi':
        return Yogi(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'msvag':
        return MSVAG(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                          weight_decay=args.weight_decay)
    # elif args.optim == 'adabound':
    #     return AdaBound(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
    #                     final_lr=args.opt_final_lr, gamma=args.opt_gamma,
    #                     weight_decay=args.weight_decay)
    else:
        print('Optimizer not found')