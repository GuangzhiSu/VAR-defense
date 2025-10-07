#!/usr/bin/env python3
"""
Infinity Circuit Breaker Training Script
Based on Infinity's train.py but with circuit breaker safety mechanisms integrated
"""

import gc
import json
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from contextlib import nullcontext
from functools import partial
from distutils.util import strtobool
from typing import List, Optional, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
import torch.distributed as tdist
import wandb

# Add circuit breaker imports
sys.path.append('../circuit-breaker_original/src')
from lorra_circuit_breaker_new import compute_infinity_circuit_breaker_loss, data_collator_infinity

sys.path.append('/home/gs285/VAR-defense/')
import infinity.utils.dist as dist
from infinity.utils.save_and_load import CKPTSaver, auto_resume
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

# Import combined args
from combined_args import CombinedArgs, init_dist_and_get_combined_args

enable_timeline_sdk = False

# Global variable for speed tracking
g_speed_ls = deque(maxlen=128)


def _rk():
    try:
        return dist.get_rank()
    except Exception:
        return 0


def dbg_print(tag: str, msg: str):
    try:
        print(f"[rk{_rk():02d}] {tag}: {msg}", flush=True)
    except Exception:
        pass


def setup_circuit_breaker_specific_args(args):
    """
    Set up circuit breaker specific attributes that are not in CombinedArgs
    """
    # Circuit breaker specific attributes that need to be added
    if not hasattr(args, 'num_examples'):
        args.num_examples = 1000
    if not hasattr(args, 'harmful_prompts_path'):
        args.harmful_prompts_path = None
    if not hasattr(args, 'sanitized_prompts_path'):
        args.sanitized_prompts_path = None
    if not hasattr(args, 'validation_ratio'):
        args.validation_ratio = 0.1
    if not hasattr(args, 'category'):
        args.category = None
    if not hasattr(args, 'circuit_breaker_alpha'):
        args.circuit_breaker_alpha = 0.1
    if not hasattr(args, 'circuit_breaker_target_layers'):
        if hasattr(args, 'target_layers') and args.target_layers:
            # Convert list to string format
            if isinstance(args.target_layers, list):
                args.circuit_breaker_target_layers = ",".join(map(str, args.target_layers))
            else:
                args.circuit_breaker_target_layers = str(args.target_layers)
        else:
            args.circuit_breaker_target_layers = "0,1,2,3,4,5"
    if not hasattr(args, 'circuit_breaker_enabled'):
        args.circuit_breaker_enabled = True
    if not hasattr(args, 'selective_layers'):
        args.selective_layers = "0,1,2,3,4,5"
    # When True, we will freeze all layers except those listed in `selective_layers`
    if not hasattr(args, 'freeze_to_selective_layers'):
        args.freeze_to_selective_layers = False
    if not hasattr(args, 'unfreeze_attention'):
        args.unfreeze_attention = False
    if not hasattr(args, 'unfreeze_output'):
        args.unfreeze_output = False
    if not hasattr(args, 'freeze_embeddings'):
        args.freeze_embeddings = True
    if not hasattr(args, 'unfreeze_layernorm'):
        args.unfreeze_layernorm = False
    if not hasattr(args, 'unfreeze_mlp'):
        args.unfreeze_mlp = False
    if not hasattr(args, 'unfreeze_cross_attention'):
        args.unfreeze_cross_attention = False
    # Scaling of retain/circuit-breaker penalties (align with LORRA)
    if not hasattr(args, 'lorra_alpha'):
        args.lorra_alpha = 1.0
    if not hasattr(args, 'coeff_schedule'):
        args.coeff_schedule = 'linear_converge'
    
    return args


def _to_ratio(x):
    # 统一把 '1:1'、'4:3'、'0.75'、1.3333 等都安全转 float
    if isinstance(x, (int, float)): 
        return float(x)
    s = str(x).strip()
    if ':' in s:
        a, b = s.split(':', 1)
        return float(a) / float(b)
    # 也兼容 '1x1' 或 '1/1' 的写法（若你的数据可能出现）
    if 'x' in s:
        a, b = s.split('x', 1)
        return float(a) / float(b)
    if '/' in s:
        a, b = s.split('/', 1)
        return float(a) / float(b)
    return float(s)

def setup_dist_if_needed(args):
    # 若用户单卡运行且没用 torchrun，则不给分布式初始化，走单机逻辑
    if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
        # 设置单卡训练参数
        setattr(args, 'rank', 0)
        setattr(args, 'world_size', 1)
        setattr(args, 'local_rank', 0)
        # 单卡也设置当前 device，保持后续代码统一
        if torch.cuda.is_available():
            torch.cuda.set_device(args.device if isinstance(args.device, int) else getattr(args.device, "index", 0))
        return

    # 分布式初始化（torchrun 会提供环境变量）
    if not dist.initialized():
        dist.init_distributed_mode(
            local_out_path=args.local_out_path,
            fork=False,
            timeout_minutes=30
        )
        # 设置分布式训练参数
        setattr(args, 'rank', dist.get_rank())
        setattr(args, 'world_size', dist.get_world_size())
        setattr(args, 'local_rank', int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(args.local_rank)

def build_everything_from_args(args: arg_util.Args, saver):
    # set seed
    args.set_initial_seed(benchmark=True)
    if args.seed is not None and not args.rand: # check the randomness
        misc.check_randomness(args)

    # setup dist
    setup_dist_if_needed(args)

    # build data
    iters_train, ld_train, ld_val = build_dataloaders(args)   
    raw_keys = list(ld_train.dataset.h_div_w_template2generator.keys())
    print(f"{raw_keys=}")
    train_h_div_w_list = [_to_ratio(k) for k in raw_keys]
    print(f"{train_h_div_w_list=}")
    args.train_h_div_w_list = train_h_div_w_list

    # load VAE
    print(f'Load vae form {args.vae_ckpt}')
    if not os.path.exists(args.vae_ckpt):
        vae_ckpt = {}
    else:
        vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')



    # build models. Note that here gpt is the causal VAR transformer which performs next scale prediciton with text guidance
    text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer(args, vae_ckpt)
    
    # IMPORTANT: import heavy package `InfinityTrainer` after the Dataloader object creation/iteration to avoid OOM
    from trainer import InfinityTrainer
    # build trainer
    trainer = InfinityTrainer(
        is_visualizer=dist.is_visualizer(), device=args.device, raw_scale_schedule=args.scale_schedule, resos=args.resos,
        vae_local=vae_local, gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp, ema_ratio=args.tema, max_it=iters_train * args.ep,
        gpt_opt=gpt_optim, label_smooth=args.ls, z_loss_ratio=args.lz, eq_loss=args.eq, xen=args.xen,
        dbg_unused=args.dbg, zero=args.zero, vae_type=args.vae_type,
        reweight_loss_by_scale=args.reweight_loss_by_scale, gpt_wo_ddp_ema=gpt_wo_ddp_ema, 
        gpt_ema=gpt_ddp_ema, use_fsdp_model_ema=args.use_fsdp_model_ema, other_args=args,
    )
    
    # auto resume from broken experiment
    auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    args.dump_log()
    if start_ep == args.ep:
        args.dump_log()
        print(f'[vgpt] AR finished ({acc_str}), skipping ...\n\n')
        return None
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    
    start_it = start_it % iters_train
    print(f"{start_it=}, {iters_train=}")
    
    del vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()
    return (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train, ld_val
    )


def build_model_optimizer(args, vae_ckpt):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from infinity.models.infinity import Infinity, MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer
    from infinity.utils.lr_control import filter_params
    from infinity.utils.load import build_vae_gpt
    
    # disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    vae_local, gpt_wo_ddp, gpt_wo_ddp_ema = build_vae_gpt(args, vae_ckpt, skip_gpt=False, device=args.model_init_device)
    del vae_ckpt
    if args.tini < 0:
        args.tini = math.sqrt(1 / gpt_wo_ddp.C / 3)
    init_weights(gpt_wo_ddp, other_std=args.tini)
    gpt_wo_ddp.special_init(aln_init=args.aln, aln_gamma_init=args.alng, scale_head=args.hd0, scale_proj=args.diva)

    # ===================== Selective-layer finetuning (freeze others) =====================
    def _parse_selective_layers(spec: str, max_depth: int):
        # Accept formats like "25-29" or "10,12,14" or mixed "1-3,6,8-9"
        idx_set = set()
        if spec is None:
            return idx_set
        for part in str(spec).split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                try:
                    lo, hi = int(a), int(b)
                except Exception:
                    continue
                if lo > hi:
                    lo, hi = hi, lo
                for i in range(lo, hi + 1):
                    if 0 <= i < max_depth:
                        idx_set.add(i)
            else:
                try:
                    i = int(part)
                except Exception:
                    continue
                if 0 <= i < max_depth:
                    idx_set.add(i)
        return sorted(idx_set)

    if getattr(args, 'freeze_to_selective_layers', False):
        depth = len(getattr(gpt_wo_ddp, 'unregistered_blocks', []))
        target_layer_indices = _parse_selective_layers(getattr(args, 'selective_layers', None), depth)

        # Freeze all parameters first
        for p in gpt_wo_ddp.parameters():
            p.requires_grad = False

        # Unfreeze only selected transformer blocks
        for idx in target_layer_indices:
            block = gpt_wo_ddp.unregistered_blocks[idx]
            for p in block.parameters():
                p.requires_grad = True

        # Optional: keep LayerNorms trainable if requested (aligns with common fine-tune practice)
        if getattr(args, 'unfreeze_layernorm', False):
            for m in gpt_wo_ddp.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

        print(f"[Selective Finetune] depth={depth}, trainable_blocks={target_layer_indices}")

    if args.rush_resume:
        print(f"{args.rush_resume=}")
        cpu_d = torch.load(args.rush_resume, 'cpu')
        if 'trainer' in cpu_d:
            state_dict = cpu_d['trainer']['gpt_fsdp']
            ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
        else:
            state_dict = cpu_d
            ema_state_dict = state_dict
        def drop_unfit_weights(state_dict):
            if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                del state_dict['word_embed.weight']
            if 'head.weight' in state_dict and (state_dict['head.weight'].shape[0] != gpt_wo_ddp.head.out_features):
                del state_dict['head.weight']
            if 'head.bias' in state_dict and (state_dict['head.bias'].shape[0] != gpt_wo_ddp.head.bias.shape[0]):
                del state_dict['head.bias']
            if state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != gpt_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape:
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]
            return state_dict
        
        gpt_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False)
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema.load_state_dict(drop_unfit_weights(ema_state_dict), strict=False)

    if args.rwe:
        gpt_wo_ddp.word_embed.weight.requires_grad = False
        torch.nn.init.trunc_normal_(gpt_wo_ddp.word_embed.weight.data, std=1.5 * math.sqrt(1 / gpt_wo_ddp.C / 3))
        if hasattr(gpt_wo_ddp.word_embed, 'bias'):
            gpt_wo_ddp.word_embed.bias.requires_grad = False
            gpt_wo_ddp.word_embed.bias.data.zero_()
    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    
    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.quant', vae_local.quantize)
    )]))
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('GPT', gpt_wo_ddp),
    )]) + '\n\n')
    
    gpt_uncompiled = gpt_wo_ddp
    gpt_wo_ddp = args.compile_model(gpt_wo_ddp, args.tfast)

    gpt_ddp_ema = None
    if args.zero and dist.initialized():
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.distributed.device_mesh import init_device_mesh

        # Ensure distributed environment is ready before FSDP initialization
        dist.barrier()

        # use mix prec: https://github.com/pytorch/pytorch/issues/76607
        if gpt_wo_ddp.num_block_chunks == 1:  # no chunks
            auto_wrap_policy = ModuleWrapPolicy([type(gpt_wo_ddp.unregistered_blocks[0]), ])
        else:
            auto_wrap_policy = ModuleWrapPolicy([MultipleLayers, ])
        
        if args.enable_hybrid_shard:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD if args.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
            assert dist.initialized()
            # Ensure distributed environment is ready
            dist.barrier()
            world_size = dist.get_world_size()
            assert world_size % args.inner_shard_degree == 0
            assert args.inner_shard_degree > 1 and args.inner_shard_degree < world_size
            device_mesh = init_device_mesh('cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree))
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
            device_mesh = None
        print(f'{">" * 45 + " " * 5} FSDP INIT with {args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)
        
        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp, 
            device_id=dist.get_local_rank(),
            sharding_strategy=sharding_strategy, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            use_orig_params=True, 
            sync_module_states=True, 
            limit_all_gathers=True,
            device_mesh=device_mesh,
        ).to(args.device)
        
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(args.device)
            gpt_ddp_ema: FSDP = FSDP(
                gpt_wo_ddp_ema, 
                device_id=dist.get_local_rank(),
                sharding_strategy=sharding_strategy, 
                mixed_precision=None,
                auto_wrap_policy=auto_wrap_policy, 
                use_orig_params=args.fsdp_orig, 
                sync_module_states=True, 
                limit_all_gathers=True,
            )
    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=args.dbg, broadcast_buffers=False)
    torch.cuda.synchronize()

    # =============== build optimizer ===============
    nowd_keys = set()
    if args.nowd >= 1:
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
        }
    if args.nowd >= 2:
        nowd_keys |= {'class_emb', 'embedding'}
    names, paras, para_groups = filter_params(gpt_ddp if args.zero else gpt_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict
    if '_' in args.ada:
        beta0, beta1 = map(float, args.ada.split('_'))
    else:
        beta0, beta1 = float(args.ada), -1
    
    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
    }[args.opt]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    if args.oeps: opt_kw['eps'] = args.oeps
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    gpt_optim = AmpOptimizer('gpt', args.fp16, opt_clz(params=para_groups, **opt_kw), gpt_ddp if args.zero else gpt_wo_ddp, args.r_accu, args.tclip, args.zero)
    del names, paras, para_groups
    
    if args.online_t5:
        print(f'Loading T5 from {args.t5_path}...')
        # 明确指定为本地路径，避免新版本 transformers 的路径验证问题
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(args.t5_path, revision=None, legacy=True)
        text_tokenizer.model_max_length = args.tlen
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(args.t5_path, torch_dtype=torch.float16)
        text_encoder.to(args.device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        [p.requires_grad_(False) for p in text_encoder.parameters()]
    else:
        text_tokenizer = text_encoder = None
    
    return text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim


def build_dataloaders(args):

    # Use circuit breaker dataset
    from cb_train_dataset_infinity import build_circuit_breaker_dataset
    
    # Create circuit breaker dataset using the new interface
    # Set default paths if not provided
    if not hasattr(args, 'harmful_prompts_path') or args.harmful_prompts_path is None:
        args.harmful_prompts_path = './data'
    if not hasattr(args, 'sanitized_prompts_path') or args.sanitized_prompts_path is None:
        args.sanitized_prompts_path = './data'
    if not hasattr(args, 'category') or args.category is None:
        args.category = 'general'
    
    print(f"Using data paths: harmful={args.harmful_prompts_path}, sanitized={args.sanitized_prompts_path}, category={args.category}")
    
    # Determine distributed context for dataset sharding
    ds_world_size = dist.get_world_size() if dist.initialized() else 1
    ds_rank = dist.get_rank() if dist.initialized() else 0

    dataset_train = build_circuit_breaker_dataset(
        args=args,
        data_path=args.data_path if hasattr(args, 'data_path') else './data',
        data_load_reso=args.data_load_reso if hasattr(args, 'data_load_reso') else 512,
        max_caption_len=args.tlen if hasattr(args, 'tlen') else 512,
        short_prob=args.short_cap_prob if hasattr(args, 'short_cap_prob') else 0.2,
        load_vae_instead_of_image=False,
        harmful_prompts_path=args.harmful_prompts_path,
        sanitized_prompts_path=args.sanitized_prompts_path,
        validation_ratio=args.validation_ratio,
        category=args.category,
        # pass DDP/FSDP info to iterable dataset to avoid cross-rank collisions
        num_replicas=ds_world_size,
        rank=ds_rank,
        dataloader_workers=1,
    )

    
    type_train_set = type(dataset_train).__name__
    vbs = round(args.batch_size * 1.5)
    print(f"{args.batch_size=}, {vbs=}", flush=True)
    ld_val = math.ceil(50000 / vbs)
    

    # Use standard DataLoader since the new dataset yields (images, captions) format
    # Create a simple generator function for the DataLoader
    def get_generator():
        if args.seed is None:
            return None          # 让 DataLoader 自己用系统随机种子
        g = torch.Generator()
        g.manual_seed(args.seed + dist.get_rank()*512)  # 多卡可区分
        return g
    
    # IMPORTANT: Avoid running heavy HF models inside DataLoader worker processes.
    # Use single-process iteration for this IterableDataset to prevent hangs/deadlocks.
    ld_train = DataLoader(
        dataset=dataset_train,
        num_workers=0,
        pin_memory=True,
        generator=get_generator(),
        batch_size=None,  # Use None for IterableDataset
        prefetch_factor=None,
    )
    
    iters_train = len(ld_train)
    print(f'len(dataloader): {len(ld_train)}, len(dataset): {len(dataset_train)}')
    if hasattr(dataset_train, 'total_samples'):
        print(f'total_samples: {dataset_train.total_samples()}')
    print(f'[dataloader] batch_size={args.batch_size}, iters_train={iters_train}, type(train_set)={type_train_set}')
    return iters_train, ld_train, ld_val


def main_train(args: arg_util.Args):
    # Initialize wandb if not already initialized and not in offline mode
    if not wandb.run and os.environ.get("WANDB_MODE") != "offline":
        try:
            # Check if WANDB_KEY is available
            if "WANDB_KEY" in os.environ:
                wandb_utils.initialize(args, entity="gs285", exp_name=args.exp_name, project_name=args.project_name)
                print("Wandb initialized successfully")
            else:
                print("WANDB_KEY not found, continuing without wandb logging...")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
    else:
        print("Wandb is in offline mode or already initialized, skipping wandb setup...")
    
    saver = CKPTSaver(dist.is_master(), eval_milestone=None)
    ret = build_everything_from_args(args, saver)
    
    if ret is None:
        return
    
    (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone,
        iters_train, ld_train, ld_val
    ) = ret
    gc.collect(), torch.cuda.empty_cache()
    
    # import heavy packages after Dataloader object creation
    from trainer import InfinityTrainer
    
    # Set enable_timeline_sdk to False for now
    enable_timeline_sdk = False
    
    # Set up logging parameters milestone
    logging_params_milestone: List[int] = np.linspace(1, args.ep, 10+1, dtype=int).tolist()
    
    ret: Tuple[
        misc.TensorboardLogger, T5TokenizerFast, T5EncoderModel, InfinityTrainer,
        int, int, str, List[Tuple[float, float]], Optional[int], Optional[DataLoader], DataLoader,
    ]

    # 使用已经设置好的 world_size，而不是直接从环境变量读取
    world_size = dist.get_world_size()
    start_time, min_L_mean, min_L_tail, max_acc_mean, max_acc_tail = time.time(), 999., 999., -1., -1.
    last_val_loss_mean, best_val_loss_mean, last_val_acc_mean, best_val_acc_mean = 999., 999., 0., 0.
    last_val_loss_tail, best_val_loss_tail, last_val_acc_tail, best_val_acc_tail = 999., 999., 0., 0.
    
    # ============================================= epoch loop begins =============================================
    ep_lg = max(1, args.ep // 10)
    epochs_loss_nan = 0
    PARA_EMB, PARA_ALN, PARA_OT, PARA_ALL = 0, 0, 0, 0
    
    for ep in range(start_ep, args.ep):
        if ep % ep_lg == 0 or ep == start_ep:
            print(f'[PT info]  from ep{start_ep} it{start_it}, acc_str: {acc_str}, diffs: {args.diffs},    =======>  bed: {args.bed}  <=======\n')
        # set epoch for dataloader
        if args.use_streaming_dataset:
            ld_train.dataset.set_epoch(ep)

        # [train one epoch]
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep,
            is_first_ep=ep == start_ep,
            start_it=start_it if ep == start_ep else 0,
            me=None,
            saver=saver,
            args=args,
            ld_or_itrt=iter(ld_train),
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
            logging_params_milestone=logging_params_milestone,
            enable_timeline_sdk=enable_timeline_sdk,
        )
        
        # [update the best loss or acc]
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        min_L_mean, max_acc_mean, max_acc_tail = min(min_L_mean, L_mean), max(max_acc_mean, acc_mean), max(max_acc_tail, acc_tail)
        if L_tail != -1:
            min_L_tail = min(min_L_tail, L_tail)
        
        # [check nan]
        epochs_loss_nan += int(not math.isfinite(L_mean))
        if (args.fp16 == 1 and epochs_loss_nan >= 2) or (args.fp16 != 1 and epochs_loss_nan >= 1):
            print(f'[rk{dist.get_rank():02d}] L_mean is {L_mean}, stopping training!', flush=True, force=True)
            sys.exit(666)
        
        # [logging]
        args.cur_phase = 'AR'
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        args.last_Lnll, args.last_Ld, args.acc_all, args.acc_real, args.acc_fake, args.last_wei_g = min_L_mean, min_L_tail, None, (None if max_acc_mean < 0 else max_acc_mean), (None if max_acc_tail < 0 else max_acc_tail), grad_norm
        if math.isfinite(args.last_wei_g) and args.last_wei_g > 4:
            args.grad_boom = 'boom'
        
        AR_ep_loss = {}
        is_val_and_also_saving = (ep + 1) % max(1, args.ep // 25) == 0 or (ep + 1) == args.ep
        if (ep + 1) < 10:
            law_stats = {
                'last_Lm': L_mean, 'best_Lm': min_L_mean, 'last_Am': acc_mean, 'best_Am': max_acc_mean,
                'last_Lt': L_tail, 'best_Lt': min_L_tail, 'last_At': acc_tail, 'best_At': max_acc_tail,
                'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
            }
        elif is_val_and_also_saving:
            if ld_val is None or isinstance(ld_val, int):    # args.nodata or args.nova
                last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = 0.666, 0.555, 5.55, 6.66, 50000, 0.001
            else:
                last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = trainer.eval_ep(ep, args, ld_val)
            
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, last_val_loss_mean), min(best_val_loss_tail, last_val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, last_val_acc_mean), max(best_val_acc_tail, last_val_acc_tail)
            AR_ep_loss['vL_mean'], AR_ep_loss['vL_tail'], AR_ep_loss['vacc_mean'], AR_ep_loss['vacc_tail'] = last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail
            print(f'  [*] [ep{ep}]  VAL {tot}  |  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Accm: {acc_mean:.2f}, Acct: {acc_tail:.2f}, cost: {cost:.2f}s')
            law_stats = {
                'last_Lm': last_val_loss_mean, 'best_Lm': best_val_loss_mean, 'last_Am': last_val_acc_mean, 'best_Am': best_val_acc_mean,
                'last_Lt': last_val_loss_tail, 'best_Lt': best_val_loss_tail, 'last_At': last_val_acc_tail, 'best_At': best_val_acc_tail,
                'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
            }
        else: 
            law_stats = None
            
        if dist.is_master() and law_stats is not None:
            stat_file = os.path.join(args.bed, 'law.stat')
            if os.path.exists(stat_file):
                with open(stat_file, 'r', encoding='utf-8') as law_fp: 
                    tag_to_epv = json.load(law_fp)
            else:
                tag_to_epv = {tag: {} for tag in law_stats.keys()}
            for tag, v in law_stats.items():
                tag_to_epv[tag][ep + 1] = v
            with open(stat_file, 'w', encoding='utf-8') as law_fp: 
                json.dump(tag_to_epv, law_fp, indent=2)
            
            # ============= LEGACY =============
            with open(os.path.join(args.bed, 'law'), 'w') as law_fp:
                json.dump({
                    'last_Lm': last_val_loss_mean, 'best_Lm': best_val_loss_mean, 'last_Am': last_val_acc_mean, 'best_Am': best_val_acc_mean,
                    'last_Lt': last_val_loss_tail, 'best_Lt': best_val_loss_tail, 'last_At': last_val_acc_tail, 'best_At': best_val_acc_tail,
                    'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
                }, law_fp, indent=2)
                
        print(f'  [*] [ep{ep}]  Lmean: {min_L_mean:.3f} ({L_mean:.3f}), Ltail {min_L_tail:.3f} ({L_tail:.3f}),  Acc m-t: {max_acc_mean:.2f} {max_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        AR_ep_loss['L_mean'], AR_ep_loss['L_tail'], AR_ep_loss['acc_mean'], AR_ep_loss['acc_tail'] = L_mean, L_tail, acc_mean, acc_tail        
        args.dump_log()
    # ============================================= epoch loop ends =============================================
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total Time: {total_time},   Lm: {min_L_mean:.3f} ({L_mean}),   Lt: {min_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats, iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    return


def train_one_ep(
    ep: int, is_first_ep: bool, start_it: int, me: misc.MetricLogger,
    saver: CKPTSaver, args: arg_util.Args, ld_or_itrt, iters_train: int, 
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer, logging_params_milestone, enable_timeline_sdk: bool,
):
    # IMPORTANT: import heavy packages after the Dataloader object creation/iteration to avoid OOM
    from trainer import InfinityTrainer
    from infinity.utils.lr_control import lr_wd_annealing
    trainer: InfinityTrainer
    
    step_cnt = 0
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    start_time = time.time()
    
    with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=20, verbose=True) as telling_dont_kill:
        last_touch = time.time()
        g_it, max_it = ep * iters_train, args.ep * iters_train
        
        doing_profiling = args.prof and ep == 0 and (args.profall or dist.is_master())
        maybe_record_function = record_function if doing_profiling else nullcontext
        trainer.gpt_wo_ddp.maybe_record_function = maybe_record_function
        
        last_t_perf = time.time()
        speed_ls: deque = g_speed_ls
        FREQ = min(args.prof_freq, iters_train//2-1)
        NVIDIA_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8))
        ranges = set([2 ** i for i in range(20)])
        if ep <= 1: ranges |= {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40}
        PRINTABLE_IT_PLUS_1 = set(FREQ*i for i in ranges)

        me = misc.MetricLogger()
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
        # ============================================= iteration loop begins =============================================
        for it, data in me.log_every(start_it, iters_train, ld_or_itrt, args.log_freq, args.log_every_iter, header):
            g_it = ep * iters_train + it

            # calling inc_step to sync the global_step
            if enable_timeline_sdk:
                ndtimeline.inc_step()

            if (it+1) % FREQ == 0:
                speed_ls.append((time.time() - last_t_perf) / FREQ)
                last_t_perf = time.time()

                if enable_timeline_sdk:
                    ndtimeline.flush()
            
            if (g_it+1) % args.save_model_iters_freq == 0:
                with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=3, verbose=True):
                    saver.sav(args=args, g_it=(g_it+1), next_ep=ep, next_it=it+1, trainer=trainer, acc_str=f'[todo]', eval_milestone=None, also_save_to=None, best_save_to=None)
            
            with maybe_record_function('before_train'):
                # The new dataset yields a rich batch dict with multiple text conditions
                if isinstance(data, dict):
                    inp = data['images'].to(args.device, non_blocking=True)
                    # Use retain text condition for main AR loss
                    text_cond_tuple = data['text_cond_tuple']
                    # Convert list of tuples to a single batched tuple expected by model
                    # We keep per-sample lens and pack features like before
                    # Build tokenizer/encoder outputs if needed
                    kv_compact_list, lens_list = [], []
                    max_len = 0
                    for t in text_cond_tuple:
                        kv_compact_i, lens_i, cu_seqlens_k_i, Ltext_i = t
                        kv_compact_list.append(kv_compact_i)
                        lens_list.extend(lens_i)
                        max_len = max(max_len, Ltext_i)
                    kv_compact = torch.cat(kv_compact_list, dim=0).to(args.device, non_blocking=True)
                    # Build cumulative seqlens for packed kv_compact across the whole batch
                    import numpy as _np
                    _lens_arr = _np.array(lens_list, dtype=_np.int32)
                    cu_seqlens_k = torch.tensor(_np.concatenate(([0], _lens_arr.cumsum())), dtype=torch.int32, device=args.device)
                    try:
                        dbg_print(
                            "retain_tuple",
                            f"lens_len={len(lens_list)} cu_seqlens_shape={tuple(cu_seqlens_k.shape)} dtype={str(cu_seqlens_k.dtype)} device={str(cu_seqlens_k.device)} first5={cu_seqlens_k[:5].tolist()} last={int(cu_seqlens_k[-1])}"
                        )
                        if cu_seqlens_k.dim() != 1 or cu_seqlens_k.numel() != (len(lens_list) + 1):
                            dbg_print("retain_tuple_err", f"INVALID cu_seqlens_k: dim={cu_seqlens_k.dim()} numel={cu_seqlens_k.numel()} expected={len(lens_list)+1}")
                    except Exception as _e:
                        dbg_print("retain_tuple_dbg_fail", f"{_e}")
                    text_cond_tuple = (kv_compact, lens_list, cu_seqlens_k, max_len)
                    # Prepare cb/val conditions (lists) for cb loss later
                    text_cond_tuple_cb = data.get('text_cond_tuple_circuit_breaker')
                    text_cond_tuple_val = data.get('text_cond_tuple_val')
                else:
                    # Fallback: old path with (images, captions)
                    inp, captions = data
                    tokens = text_tokenizer(text=captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')
                    input_ids = tokens.input_ids.cuda(non_blocking=True)
                    mask = tokens.attention_mask.cuda(non_blocking=True)
                    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
                    lens: List[int] = mask.sum(dim=-1).tolist()
                    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                    Ltext = max(lens)
                    kv_compact = []
                    for len_i, feat_i in zip(lens, text_features.unbind(0)):
                        kv_compact.append(feat_i[:len_i])
                    kv_compact = torch.cat(kv_compact, dim=0)
                    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
                    inp = inp.to(args.device, non_blocking=True)
                
                if it > start_it + 10:
                    telling_dont_kill.early_stop()
                
                # [logging]
                args.cur_it = f'{it+1}/{iters_train}'
                args.last_wei_g = me.meters['tnm'].median
                if dist.is_local_master() and (it >= start_it + 10) and (time.time() - last_touch > 90):
                    _, args.remain_time, args.finish_time = me.iter_time.time_preds(max_it - g_it + (args.ep - ep) * 15)      # +15: other cost
                    args.dump_log()
                    last_touch = time.time()
                
                # [schedule learning rate]
                wp_it = args.wp * iters_train
                min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.gpt_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
                
                # [get scheduled hyperparameters]
                progress = g_it / (max_it - 1)
                clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1
                
                stepping = (g_it + 1) % args.ac == 0
                step_cnt += int(stepping)
            
            with maybe_record_function('in_training'):
                # Compute optional retain/cb losses using reference of text conditions (lightweight proxy)
                circuit_breaker_loss = None
                if isinstance(data, dict) and ('text_cond_tuple_circuit_breaker' in data):
                    # Use student logits on retain and cb prompts; encourage similarity to retain, dissimilarity to cb
                    try:
                        # Student logits on retain prompts (already have text_cond_tuple batched)
                        # For cb/val, take first item as proxy to avoid heavy duplication
                        # You can extend to full-batch if needed
                        # Debug retain tuple before forward
                        try:
                            kv_r, lens_r, cu_r, L_r = text_cond_tuple
                            dbg_print(
                                "retain_forward",
                                f"kv_shape={tuple(kv_r.shape)} lens_len={len(lens_r)} cu_shape={tuple(cu_r.shape)} cu_dtype={str(cu_r.dtype)} L={int(L_r)}"
                            )
                        except Exception as _e:
                            dbg_print("retain_forward_dbg_fail", f"{_e}")
                        logits_ret = trainer.logits_from_inputs(inp, text_cond_tuple, args, use_student=True)
                        if text_cond_tuple_cb and isinstance(text_cond_tuple_cb, list):
                            # text_cond_tuple_cb may carry prebuilt (kv, lens, cu_seqlens, L) for CB prompts.
                            # Some datasets provide cu_seqlens that are not 1D of length (num_seqs + 1).
                            # Repack to ensure cu_seqlens_k has valid shape (num_seqs + 1)
                            kv_c_list, lens_c, cu_c, L_c = text_cond_tuple_cb[0]
                            if isinstance(lens_c, torch.Tensor):
                                lens_c_list = lens_c.tolist()
                            else:
                                lens_c_list = list(lens_c)
                            # Normalize cu_c to 1D int32 cumulative lengths
                            try:
                                cu_c_1d = cu_c.view(-1)
                            except Exception:
                                cu_c_1d = cu_c
                            need_repack = (cu_c_1d.dtype != torch.int32) or (cu_c_1d.dim() != 1) or (cu_c_1d.numel() != (len(lens_c_list) + 1))
                            if need_repack:
                                _lens_arr_c = _np.array(lens_c_list, dtype=_np.int32)
                                cu_c_1d = torch.tensor(_np.concatenate(([0], _lens_arr_c.cumsum())), dtype=torch.int32, device=args.device)
                            else:
                                cu_c_1d = cu_c_1d.to(device=args.device, dtype=torch.int32)
                            cb_tuple = (kv_c_list.to(args.device, non_blocking=True), lens_c_list, cu_c_1d, int(L_c))
                            try:
                                dbg_print(
                                    "cb_tuple",
                                    f"kv_shape={tuple(cb_tuple[0].shape)} lens_len={len(cb_tuple[1])} cu_shape={tuple(cb_tuple[2].shape)} cu_dtype={str(cb_tuple[2].dtype)} L={cb_tuple[3]} need_repack={need_repack}"
                                )
                                if cb_tuple[2].dim() != 1 or cb_tuple[2].numel() != (len(cb_tuple[1]) + 1):
                                    dbg_print("cb_tuple_err", f"INVALID cu_seqlens_k: dim={cb_tuple[2].dim()} numel={cb_tuple[2].numel()} expected={len(cb_tuple[1])+1}")
                            except Exception as _e:
                                dbg_print("cb_tuple_dbg_fail", f"{_e}")
                            logits_cb = trainer.logits_from_inputs(inp, cb_tuple, args, use_student=True)
                        else:
                            logits_cb = None

                        # Retain loss: encourage consistency on retain vs itself (identity); implement as small L2 on logits history if needed
                        retain_loss = 0.0 * logits_ret.mean()

                        # Circuit breaker loss: discourage alignment to cb logits by maximizing similarity negativity
                        if logits_cb is not None and logits_cb.shape == logits_ret.shape:
                            # Normalize and penalize positive cosine similarity
                            ret_n = torch.nn.functional.normalize(logits_ret.float(), dim=-1)
                            cb_n = torch.nn.functional.normalize(logits_cb.float(), dim=-1)
                            inner = (ret_n * cb_n).sum(dim=-1)  # [B, L]
                            cb_loss = torch.relu(inner).mean()
                        else:
                            cb_loss = None

                        if cb_loss is not None:
                            # ===== coefficient schedule (match lorra_circuit_breaker.py) =====
                            if args.coeff_schedule == 'linear_converge':
                                progress = g_it / max(1, (max_it - 1))
                            else:
                                progress = g_it / max(1, (max_it - 1))
                            scheduled_coeff = progress
                            retain_coeff = args.lorra_alpha * scheduled_coeff
                            cb_coeff = args.lorra_alpha * (1 - scheduled_coeff)
                            circuit_breaker_loss = retain_coeff * retain_loss + cb_coeff * cb_loss
                            try:
                                dbg_print("cb_loss", f"ret_loss={float(retain_loss)} cb_loss={float(cb_loss)} coeffs=(ret={retain_coeff:.4f}, cb={cb_coeff:.4f}) total={float(circuit_breaker_loss)}")
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Warning: CB loss computation skipped due to: {e}")
                        dbg_print("cb_exception", f"{repr(e)}")

                # Standard training step, optionally with circuit_breaker_loss added inside trainer
                grad_norm_t, scale_log2_t = trainer.train_step(
                    ep=ep, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=clip_decay_ratio,
                    metric_lg=me, 
                    logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone), 
                    inp_B3HW=inp, 
                    text_cond_tuple=text_cond_tuple,
                    args=args,
                    circuit_breaker_loss=circuit_breaker_loss,
                )
                
                # Update metrics
                me.update(tnm=grad_norm_t)
                # Note: stats will be updated by the trainer, we just update the grad norm here
                
                # Log progress
                if it % max(1, args.log_freq // 2) == 0:
                    print(f'{header} [{it+1:4d}/{iters_train}] Lm: {me.meters["Lm"].median:.3f}, Lt: {me.meters["Lt"].median:.3f}, Accm: {me.meters["Accm"].median:.2f}, Acct: {me.meters["Acct"].median:.2f}, Grad: {me.meters["tnm"].median:.2f}', flush=True)
        
        # Return training statistics
        stats = {
            'Lm': me.meters['Lm'].median,
            'Lt': me.meters['Lt'].median,
            'Accm': me.meters['Accm'].median,
            'Acct': me.meters['Acct'].median,
            'tnm': me.meters['tnm'].median,
        }
        
        # Calculate time statistics
        total_time = time.time() - start_time
        remain_time = f'{total_time / (ep + 1) * (args.ep - ep - 1) / 60:.1f}m'
        finish_time = f'{(time.time() + total_time / (ep + 1) * (args.ep - ep - 1)) / 3600:.1f}h'
        
        return stats, (total_time, remain_time, finish_time)


def main():     
    # Use combined args system
    args = init_dist_and_get_combined_args()
    
    # Set up circuit breaker specific arguments
    args = setup_circuit_breaker_specific_args(args)
    
    # Print configuration
    print("="*60)
    print("INFINITY CIRCUIT BREAKER TRAINING - SELECTIVE LAYER FINE-TUNING")
    print(f"Circuit Breaker Alpha: {args.circuit_breaker_alpha}")
    print(f"Circuit Breaker Target Layers: {args.circuit_breaker_target_layers}")
    print(f"Circuit Breaker Enabled: {args.circuit_breaker_enabled}")
    print(f"Number of Examples: {args.num_examples}")
    print(f"Model Name/Path: {args.model_name_or_path}")
    print(f"Harmful Prompts Path: {args.harmful_prompts_path}")
    print(f"Sanitized Prompts Path: {args.sanitized_prompts_path}")
    print(f"Validation Ratio: {args.validation_ratio}")
    print(f"Category: {args.category}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")
    print(f"VAE Checkpoint: {args.vae_ckpt}")
    print(f"Rush Resume: {args.rush_resume}")
    print(f"Selective Layers: {args.selective_layers}")
    print("="*60)
    
    main_train(args)


if __name__ == "__main__":
    main() 