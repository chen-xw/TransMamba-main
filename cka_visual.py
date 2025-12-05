# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import os
import torch
import torch.backends.cudnn as cudnn
import json
import wandb
# from logger import Logger
from torch.utils.tensorboard import SummaryWriter 
from mmcv import Config, DictAction

from config import get_config

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from Plainmamba.plain_mamba import PlainMamba

from mmcls.models import build_classifier

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator
from torch.utils.data import Subset

from vmamba import VSSM

import models
import models_v2

import utils


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--model-type', default='PlainMamba', type=str,choices=['PlainMamba', 'Mamba', 'DeiT', 'DeiT-Finetune'])
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixpu and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=2.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters   /data3/xiuwei/Datasets/imagenet100  /data3/xiuwei/Datasets
    parser.add_argument('--data-path', default='/data2/xiuwei/Datasets/imagenet100', type=str,   
                        help='dataset path')
    parser.add_argument('--data-set', default='CIFAR', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='checkpoint',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

# still on developing...
def build_vssm_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        return model

    return None

def build_PlainMamba(args):
    layer_cfgs = {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
    model = PlainMamba(
        img_size=args.input_size,
        patch_size=16,
        embed_dims=768, # 192 384, 768
        num_layers=12, # 12 12 12
        num_classes=100,
        num_convs_patch_embed=2,
        layers_with_dwconv=[], # userful for L1 model
        with_pos_embed=True,
        layer_cfgs=layer_cfgs
    )
    return model


def main(args):
    utils.init_distributed_mode(args)

    # wan = wandb.init(project='DeiT_{}_{}_{}'.format(args.model_type, args.data_set,args.input_size),
    #            name='Semantic Align',
    #            group='DDP',
    #         #    config=config
    #            )

    print(args)

    config = get_config(args)
    writer = SummaryWriter('./logs/{}_{}_{}'.format(args.model_type, args.data_set,args.input_size))

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)


    # # Only for 1/10 数据量
    # class_counts = 2 # 每个类选择十分之一的数据
    # targets = np.array(dataset_train.targets)
    # num_classes = len(np.unique(targets))

    # # 保存每个类样本的索引
    # class_indices = {i: np.where(targets == i)[0] for i in range(num_classes)}

    # # 选取每个类的class_counts个样本
    # selected_indices = []
    # for i in range(num_classes):
    #     class_sample = np.random.choice(class_indices[i], size=len(class_indices[i]) // class_counts, replace=False)
    #     selected_indices.extend(class_sample)
    
    # # 使用Subset来构造新数据集
    
    # subset_trained = Subset(dataset_train, selected_indices)
    # dataset_train = subset_trained

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")

    model_deit = None
    # Teacher Model
    model_deit = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size
        )

    # Student Model
    if args.model_type =='PlainMamba':
        model = build_PlainMamba(args)
    elif args.model_type =='Mamba':
        model = build_vssm_model(config, False)  # is_pretrain = False
    elif args.model_type =='DeiT' or args.model_type =='DeiT-Finetune':
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size
        )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # # interpolate position embedding
        # pos_embed_checkpoint = checkpoint_model['pos_embed']
        # embedding_size = pos_embed_checkpoint.shape[-1]
        # num_patches = model.patch_embed.num_patches
        # num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # # height (== width) for the checkpoint position embedding
        # orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # # height (== width) for the new position embedding
        # new_size = int(num_patches ** 0.5)
        # # class_token and dist_token are kept unchanged
        # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # # only the position tokens are interpolated
        # pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        # pos_tokens = torch.nn.functional.interpolate(
        #     pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        # checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    # teacher_model = None
    # if args.distillation_type != 'none':
    #     assert args.teacher_path, 'need to specify teacher-path when using distillation'
    #     print(f"Creating teacher model: {args.teacher_model}")
    #     teacher_model = create_model(
    #         args.teacher_model,
    #         pretrained=False,
    #         num_classes=args.nb_classes,
    #         global_pool='avg',
    #     )
    #     if args.teacher_path.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.teacher_path, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.teacher_path, map_location='cpu')
    #     teacher_model.load_state_dict(checkpoint['model'])
    #     teacher_model.to(device)
    #     teacher_model.eval()
    # from IPython import embed
    # embed()
    checkpoint_model = torch.load('/data2/xiuwei/Code/deit-main/checkpoint/PlainMamba_CIFAR_224_checkpoint_best.pth')
    model.load_state_dict(checkpoint_model['model'])
    

    # Teacher Model checkpoint
    checkpoint_deit = torch.load('/data2/xiuwei/Code/deit-main/checkpoint/DeiT-Finetune_CIFAR_224_checkpoint_best.pth')
    model_deit.load_state_dict(checkpoint_deit['model'], strict = False)

    # update BN layer
    model_deit.eval()
    model_deit.to(device)
    from torch_cka import CKA
    cka = CKA(model, model_deit,
              model1_name='Mamba',
              model2_name='DeIT',
              model1_layers=['layers.11'],
              model2_layers=['blocks.11'])
    cka.compare(data_loader_train)
    cka.plot_results()




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mamba_based training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# CUDA_VISIBLE_DEVICES=4,6 python -m torch.distributed.launch --master_port=14561 --nproc_per_node=2 --use_env main.py > output/Mamba_CIFAR224.out 2>&1 &
   # CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port=51261 --nproc_per_node=2 --use_env main.py > output/DeiT_CIFAR224.out 2>&1 &
    # python main.py --eval --resume /data3/xiuwei/Code/deit-main/checkpoint/deit_base_patch16_224-b5f2ef4d.pth --data-path /data3/xiuwei/Datasets
       # CUDA_VISIBLE_DEVICES=4,6 python -m torch.distributed.launch --master_port=14521 --nproc_per_node=2 --use_env main.py > output/PlainMamba_CIFAR224.out 2>&1 &
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=12461 --nproc_per_node=4 --use_env main.py --distillation-type soft > output/PlainMamba_Small_CIFAR224.out 2>&1 &
    # CUDA_VISIBLE_DEVICES=4,6 python -m torch.distributed.launch --master_port=15689 --nproc_per_node=2 --use_env main.py --finetune  '/data2/xiuwei/Code/deit-main/checkpoint/deit_tiny_patch16_224-a1311bcf.pth' > output/Deit_Finetune_CIFAR224_2.out 2>&1 &

# CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port=12141 --nproc_per_node=2 --use_env main.py  --finetune  '/data2/xiuwei/Code/deit-main/checkpoint/deit_base_patch16_224-b5f2ef4d.pth' > output/DeiT_finetune_base_CIFAR224.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port=16931 --nproc_per_node=2 --use_env main.py > output/PlainMamba_Base4_CIFAR224.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=12861 --nproc_per_node=4 --use_env main.py --distillation-type soft > output/PlainMamba_Small_IMNT224.out 2>&1 &



# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port=12461 --nproc_per_node=2 --use_env main.py --distillation-type soft > output_align/PlainMamba_tiny_teacher-small_CIFAR224_zero_padding_zeromean_unitvarianceonlyMamba_teacherbnupdate.out 2>&1 &