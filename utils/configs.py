import argparse

from torchvision.transforms import transforms

from datasets import PascalPartSegmentation


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/pascalpart',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='pascalpart',
                        choices=['pascalpart'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=7,
                        help="num classes (default: None)")

    # Model Options
    parser.add_argument("--model", type=str, default='ACE2P_resnet101',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50', 'ACE2P_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101', 'ACE2P_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--use_abn", action='store_true', default=False,
                        help="if true ace2p model will use active batchnorm instead of batchnorm")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='SCP',
                        choices=['MSE', 'CE', 'FL', 'F1', 'SCP'], help="loss type (default: False)")
    parser.add_argument("--loss_weights", type=list,
                        default=[0.03530634, 0.15666913, 0.15524384, 0.16220391, 0.16311258, 0.16293769, 0.16452651],
                        help="loss weights for classes (default: None)")
    parser.add_argument("--gpu_ids", type=str, default='0',
                        help="GPU IDs")
    parser.add_argument("--random_seed", type=int, default=149,
                        help="random seed (default: 149)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")

    # Optimizer Options
    parser.add_argument('--optimizer', default='adabelief', type=str, help='Optimizer',
                        choices=['sgd', 'adam', 'adamw', 'adabelief', 'yogi', 'msvag', 'radam', 'fromage', ])
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--opt_eps', default=1e-8, type=float, help='eps for var adam')
    parser.add_argument('--opt_momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--opt_beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--opt_beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay for optimizers (default: 1e-4)')
    # parser.add_argument('--reset', action = 'store_true',
    #                     help='whether reset optimizer at learning rate decay')

    # Extra Train Options
    parser.add_argument("--use_mixup", action='store_true', default=False,
                        help='to either use mixup or not')
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                        help='the alpha parameter used in mixup (default: 0.2)')
    parser.add_argument("--use_mixup_mwh", action='store_true', default=False,
                        help='if using mixup, you can choose to train with Mixup Without Hesitation method')
    parser.add_argument("--mwh_stages", type=list, default=[0.6, 0.9],
                        help='the percent of the max iteration to be set for each stage in mwh. (it has 3 stages)')

    # SCHP Train Options
    parser.add_argument("--use_schp", action='store_true', default=True,
                        help='to either use SCHP or not')
    parser.add_argument("--schp_start", type=int, default=10,
                        help='SCHP start epoch')
    parser.add_argument("--cycle_epochs", type=int, default=2,
                        help='SCHP cyclical epoch')
    parser.add_argument("--schp_ckpt", default=None, type=str,
                        help="restore schl model from checkpoint")


    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    if opts.dataset == 'pascalpart':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        train_dst = PascalPartSegmentation(root=opts.data_root, split='train', crop_size=[512, 512], scale_factor=0.25,
                                           rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=transform)
        val_dst = PascalPartSegmentation(root=opts.data_root, split='val', crop_size=[512, 512], scale_factor=0,
                                         rotation_factor=0, ignore_label=255, flip_prob=0, transform=transform)

    else:
        raise Exception("Wrong dataset given. supported choices: pascalpart")

    return train_dst, val_dst
