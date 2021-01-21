from utils.optimizers import Fromage, RAdam, AdamW, AdaBelief, Yogi, MSVAG
from torch import optim


def create_optimizer(args, model_params):
    args.optimizer = args.optimizer.lower()
    print(f"Optimizer: {args.optimizer}")
    if args.optimizer == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.opt_momentum,
                         weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                          weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.optimizer == 'fromage':
        return Fromage(model_params, args.lr)
    elif args.optimizer == 'radam':
        return RAdam(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                     weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.optimizer == 'adamw':
        return AdamW(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                     weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.optimizer == 'adabelief':
        return AdaBelief(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                         weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.optimizer == 'yogi':
        return Yogi(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                    weight_decay=args.weight_decay)
    elif args.optimizer == 'msvag':
        return MSVAG(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
                     weight_decay=args.weight_decay)
    # elif args.optim == 'adabound':
    #     return AdaBound(model_params, args.lr, betas=(args.opt_beta1, args.opt_beta2),
    #                     final_lr=args.opt_final_lr, gamma=args.opt_gamma,
    #                     weight_decay=args.weight_decay)
    else:
        print('Optimizer not found')
