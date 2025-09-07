import json
import math
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict, deque
from typing import Optional, Union, List, Dict, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from tap import Tap
import transformers

import infinity.utils.dist as dist


class CombinedArgs(Tap):
    # =============== Infinity 原有参数 ===============
    local_out_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')
    data_path: str = ''
    bed: str = ''
    vae_ckpt: str = ''
    exp_name: str = ''
    ds: str = 'oi'
    model: str = ''
    short_cap_prob: float = 0.2
    project_name: str = 'Infinity'
    tf32: bool = True
    auto_resume: bool = True
    rush_resume: str = ''
    nowd: int = 1
    enable_hybrid_shard: bool = False
    inner_shard_degree: int = 1
    zero: int = 0
    buck: str = 'chunk'
    fsdp_orig: bool = True
    enable_checkpointing: str = None
    pad_to_multiplier: int = 1
    log_every_iter: bool = False
    checkpoint_type: str = 'torch'
    seed: int = None
    rand: bool = True
    device: str = 'cpu'
    task_id: str = '2493513'
    trial_id: str = '7260554'
    robust_run_id: str = '00'
    ckpt_trials: List[str] = []
    real_trial_id: str = '7260552'
    chunk_nodes: int = None
    is_master_node: bool = None
    
    # dir
    log_txt_path: str = ''
    t5_path: str = ''
    online_t5: bool = True
    
    # GPT
    sdpa_mem: bool = True
    tfast: int = 0
    model_alias: str = 'b'
    rms: bool = False
    aln: float = 1e-3
    alng: float = -1
    saln: bool = False
    haln: bool = True
    nm0: bool = False
    tau: float = 1
    cos: bool = True
    swi: bool = False
    dp: float = -1
    drop: float = 0.0
    hd: int = 0
    ca_gamma: float = -1
    diva: int = 1
    hd0: float = 0.02
    dec: int = 1
    cum: int = 3
    rwe: bool = False
    tp: float = 0.0
    tk: float = 0.0
    tini: float = 0.02
    cfg: float = 0.1
    rand_uncond = False
    ema: float = 0.9999
    tema: float = 0
    fp16: int = 0
    fuse: bool = False
    fused_norm: bool = False
    flash: bool = False
    xen: bool = False
    use_flex_attn: bool = False
    stable: bool = False
    gblr: float = 1e-4
    dblr: float = None
    tblr: float = 6e-4
    glr: float = None
    dlr: float = None
    tlr: float = None
    gwd: float = 0.005
    dwd: float = 0.0005
    twd: float = 0.005
    gwde: float = 0
    dwde: float = 0
    twde: float = 0
    ls: float = 0.0
    lz: float = 0.0
    eq: int = 0
    ep: int = 100
    wp: float = 0
    wp0: float = 0.005
    wpe: float = 0.3
    sche: str = ''
    log_freq: int = 50
    gclip: float = 6.
    dclip: float = 6.
    tclip: float = 2.
    cdec: bool = False
    opt: str = 'adamw'
    ada: str = ''
    dada: str = ''
    oeps: float = 0
    afuse: bool = True
    
    # data
    pn: str = ''
    scale_schedule: tuple = None
    patch_size: int = None
    resos: tuple = None
    data_load_reso: int = None
    workers: int = 0
    lbs: int = 0
    bs: int = 8  # 设置默认batch size
    batch_size: int = 8  # 设置默认batch size
    glb_batch_size: int = 8  # 设置默认global batch size
    ac: int = 1
    r_accu: float = 1.0
    norm_eps: float = 1e-6
    tlen: int = 512
    Ct5: int = 2048
    use_bit_label: int = 1
    bitloss_type: str = 'mean'
    dynamic_resolution_across_gpus: int = 1
    enable_dynamic_length_prompt: int = 0
    use_streaming_dataset: int = 0
    iterable_data_buffersize: int = 90000
    save_model_iters_freq: int = 1000
    noise_apply_layers: int = -1
    noise_apply_strength: float = -1
    noise_apply_requant: int = 1
    rope2d_each_sa_layer: int = 0
    rope2d_normalized_by_hw: int = 1
    use_fsdp_model_ema: int = 0
    add_lvl_embeding_only_first_block: int = 1
    reweight_loss_by_scale: int = 0
    always_training_scales: int = 100
    vae_type: int = 14  
    fake_vae_input: bool = False
    model_init_device: str = 'cuda'
    prefetch_factor: int = 2
    apply_spatial_patchify: int = 1
    debug_bsc: int = 0
    task_type: str = 't2i'
    
    # =============== Circuit-breaker 参数 ===============
    # LorraArguments
    target_layers: str = ""
    transform_layers: str = ""
    lorra_alpha: float = 5
    trainsets: List[str] = None
    valsets: List[str] = None
    adv_string: str = ""
    full_layers: bool = False
    
    # Circuit-breaker specific parameters
    harmful_prompts_path: str = None
    sanitized_prompts_path: str = None
    validation_ratio: float = 0.1
    category: str = "general"  # 设置默认category
    
    # LoraArguments
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Union[List[str], str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    
    # ModelArguments
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"
    adapter_name_or_path: str = None
    use_lora: bool = False
    
    # TrainingArguments (继承自 transformers.TrainingArguments)
    cache_dir: str = None
    optim: str = "adamw_torch"
    model_max_length: int = 512
    grouped_to_max_length: bool = False
    use_refusal_retain: bool = True
    sc_train_subset: List[str] = None
    log_every: int = 10
    sc_train_seq_type: str = 'all_text'
    coeff_schedule: str = 'linear_converge'
    sc_loss_type: str = 'orig_act_dotprod'
    
    # =============== 自动设置的参数 ===============
    branch: str = '[unknown]'
    commit_id: str = ''
    commit_msg: str = ''
    cmd: str = ''
    tag: str = 'UK'
    acc_all: float = None
    acc_real: float = None
    acc_fake: float = None
    last_Lnll: float = None
    last_L1: float = None
    last_Ld: float = None
    last_wei_g: float = None
    grad_boom: str = None
    diff: float = None
    diffs: str = ''
    diffs_ema: str = None
    ca_performance: str = ''
    cur_phase: str = ''
    cur_it: str = ''
    cur_ep: str = ''
    remain_time: str = ''
    finish_time: str = ''
    iter_speed: float = None
    img_per_day: float = None
    max_nvidia_smi: float = 0
    max_memory_allocated: float = None
    max_memory_reserved: float = None
    num_alloc_retries: int = None
    MFU: float = None
    HFU: float = None
    
    # =============== 调试参数 ===============
    dbg_modified: bool = False
    dbg_ks: bool = False
    dbg_ks_last = None
    dbg_ks_fp = None
    dbg: bool = 'KEVIN_LOCAL' in os.environ
    ks: bool = False
    nodata: bool = False
    nodata_tlen: int = 320
    nova: bool = False
    prof: int = 0
    prof_freq: int = 50
    tos_profiler_file_prefix: str = 'vgpt_default/'
    profall: int = 0
    v_seed: int = 0
    g_seed: int = 0
    
    def configure(self) -> None:
        """Configure Tap to handle complex types and provide better error messages."""
        from typing import get_origin, get_args, Union, Optional
        
        # Define type conversion functions for Union types
        def to_lora_target_modules(value: str) -> Union[List[str], str]:
            """Convert string to lora_target_modules (Union[List[str], str])."""
            if ',' in value:
                return [x.strip() for x in value.split(',')]
            return value
        
        def to_trainsets(value: str) -> Optional[List[str]]:
            """Convert string to trainsets (Optional[List[str]])."""
            if value.lower() in ('none', 'null', ''):
                return None
            if '#' in value:
                return [x.strip() for x in value.split('#')]
            return [value] if value else None
        
        def to_valsets(value: str) -> Optional[List[str]]:
            """Convert string to valsets (Optional[List[str]])."""
            if value.lower() in ('none', 'null', ''):
                return None
            if '#' in value:
                return [x.strip() for x in value.split('#')]
            return [value] if value else None
        
        def to_target_layers(value: str) -> Optional[List[int]]:
            """Convert string to target_layers (Optional[List[int]])."""
            if value.lower() in ('none', 'null', ''):
                return None
            if ',' in value:
                try:
                    return [int(l.strip()) for l in value.split(',') if l.strip()]
                except ValueError:
                    raise ValueError(f"Cannot parse target_layers '{value}' as integers")
            return [int(value)] if value else None
        
        def to_transform_layers(value: str) -> Optional[List[int]]:
            """Convert string to transform_layers (Optional[List[int]])."""
            if value.lower() in ('none', 'null', ''):
                return None
            if ',' in value:
                try:
                    return [int(l.strip()) for l in value.split(',') if l.strip()]
                except ValueError:
                    raise ValueError(f"Cannot parse transform_layers '{value}' as integers")
            return [int(value)] if value else None
        
        # Add explicit type functions for Union types as required by Tap
        self.add_argument('--lora_target_modules', type=to_lora_target_modules)
        self.add_argument('--trainsets', type=to_trainsets)
        self.add_argument('--valsets', type=to_valsets)
        self.add_argument('--target_layers', type=to_target_layers)
        self.add_argument('--transform_layers', type=to_transform_layers)
        
        # Process specific fields after parsing
        self._process_special_fields()
    
    def _process_special_fields(self):
        """Process special fields that need custom parsing after argument parsing."""
        # Process lora_target_modules
        if hasattr(self, 'lora_target_modules') and isinstance(self.lora_target_modules, str):
            if ',' in self.lora_target_modules:
                self.lora_target_modules = [x.strip() for x in self.lora_target_modules.split(',')]
        
        # Process trainsets
        if hasattr(self, 'trainsets') and isinstance(self.trainsets, str):
            if '#' in self.trainsets:
                self.trainsets = [x.strip() for x in self.trainsets.split('#')]
        
        # Process valsets
        if hasattr(self, 'valsets') and isinstance(self.valsets, str):
            if '#' in self.valsets:
                self.valsets = [x.strip() for x in self.valsets.split('#')]
        
        # Process target_layers
        if hasattr(self, 'target_layers') and isinstance(self.target_layers, str):
            if ',' in self.target_layers:
                try:
                    self.target_layers = [int(l.strip()) for l in self.target_layers.split(',') if l.strip()]
                except ValueError:
                    print(f"Warning: Could not parse target_layers '{self.target_layers}' as integers")
        
        # Process transform_layers
        if hasattr(self, 'transform_layers') and isinstance(self.transform_layers, str):
            if ',' in self.transform_layers:
                try:
                    self.transform_layers = [int(l.strip()) for l in self.transform_layers.split(',') if l.strip()]
                except ValueError:
                    print(f"Warning: Could not parse transform_layers '{self.transform_layers}' as integers")
    
    @property
    def is_vae_visualization_only(self) -> bool:
        return self.v_seed > 0
    
    @property
    def is_gpt_visualization_only(self) -> bool:
        return self.g_seed > 0
    
    @property
    def gpt_training(self):
        return len(self.model) > 0
    
    def set_initial_seed(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            seed = self.seed + (dist.get_rank()*512 if self.rand else 0)
            torch.backends.cudnn.deterministic = True
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _scan_bad_defaults(obj):
        """Scan for bad default values that can't be deep copied.
        
        Args:
            obj: Either a class or an instance to scan
            
        Returns:
            List of tuples containing (name, type, exception) for problematic defaults
        """
        import copy
        from dataclasses import is_dataclass
        from typing import get_origin, get_args, Union, Optional
        bad = []
        
        def _check_value_deepcopyable(name, value, expected_type=None):
            """Check if a value can be deep copied and validate against expected type."""
            try:
                # Try to deep copy the value
                copied_value = copy.deepcopy(value)
                
                # If we have an expected type, validate the copied value
                if expected_type is not None:
                    if not _is_valid_type(copied_value, expected_type):
                        bad.append((name, type(value), f"Value {copied_value} of type {type(value)} is not compatible with expected type {expected_type}"))
                        
            except Exception as e:
                bad.append((name, type(value), e))
        
        def _is_valid_type(value, expected_type):
            """Check if a value is compatible with the expected type annotation."""
            if expected_type is None:
                return True
                
            origin = get_origin(expected_type)
            args = get_args(expected_type)
            
            # Handle Union types (including Optional which is Union[T, None])
            if origin is Union:
                return any(_is_valid_type(value, arg) for arg in args)
            
            # Handle None specifically for Optional types
            if expected_type == type(None) or expected_type == Optional:
                return value is None
            
            # Handle basic types
            if expected_type in (str, int, float, bool):
                return isinstance(value, expected_type)
            
            # Handle List types
            if origin is list and len(args) == 1:
                if not isinstance(value, list):
                    return False
                return all(_is_valid_type(item, args[0]) for item in value)
            
            # Handle Dict types
            if origin is dict and len(args) == 2:
                if not isinstance(value, dict):
                    return False
                return all(_is_valid_type(k, args[0]) and _is_valid_type(v, args[1]) 
                          for k, v in value.items())
            
            # Handle Tuple types
            if origin is tuple:
                if not isinstance(value, tuple):
                    return False
                if len(args) == 0:  # Tuple without type args
                    return True
                if len(args) == 1 and args[0] == ():  # Empty tuple
                    return len(value) == 0
                if len(args) == 2 and args[1] == ...:  # Variable length tuple
                    return all(_is_valid_type(item, args[0]) for item in value)
                else:  # Fixed length tuple
                    if len(value) != len(args):
                        return False
                    return all(_is_valid_type(item, arg) for item, arg in zip(value, args))
            
            # For other types, just check if it's an instance
            try:
                return isinstance(value, expected_type)
            except TypeError:
                # If isinstance fails, assume it's valid
                return True
        
        # If it's a class, scan class variables
        if isinstance(obj, type):
            for name, value in vars(obj).items():
                # Skip methods and special attributes
                if callable(value) or name.startswith('_'):
                    continue
                
                # Get the type annotation if available
                type_annotation = None
                if hasattr(obj, '__annotations__') and name in obj.__annotations__:
                    type_annotation = obj.__annotations__[name]
                
                _check_value_deepcopyable(name, value, type_annotation)
            
            # If it's a dataclass, also scan dataclass fields
            if is_dataclass(obj):
                for f in obj.__dataclass_fields__.values():
                    try:
                        copied_default = copy.deepcopy(f.default)
                        # Check if the copied value matches the field type
                        if hasattr(f, 'type'):
                            if not _is_valid_type(copied_default, f.type):
                                bad.append((f.name, type(f.default), f"Default value {copied_default} is not compatible with field type {f.type}"))
                    except Exception as e:
                        bad.append((f.name, type(f.default), e))
        
        # If it's an instance, scan instance attributes
        else:
            for name, value in vars(obj).items():
                # Skip methods and special attributes
                if callable(value) or name.startswith('_'):
                    continue
                
                # Get the type annotation from the class if available
                type_annotation = None
                if hasattr(obj.__class__, '__annotations__') and name in obj.__class__.__annotations__:
                    type_annotation = obj.__class__.__annotations__[name]
                
                _check_value_deepcopyable(name, value, type_annotation)
        
        return bad
    
    @classmethod
    def scan_class_defaults(cls):
        """Scan class defaults for problematic values that can't be deep copied.
        
        Returns:
            List of tuples containing (name, type, exception) for problematic class defaults
        """
        return cls._scan_bad_defaults(cls)
    

    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed + dist.get_rank()*512)
        return g

    def compile_model(self, m, fast):
        if fast == 0:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m
    
    def dump_log(self):
        if not dist.is_local_master():
            return
        nd = {'is_master': dist.is_visualizer()}
        r_trial, trial = str(self.real_trial_id), str(self.trial_id)
        for k, v in {
            'name': self.exp_name, 'tag': self.tag, 'cmd': self.cmd, 'commit': self.commit_id, 'branch': self.branch,
            'Lnll': self.last_Lnll, 'L1': self.last_L1,
            'Ld': self.last_Ld,
            'acc': self.acc_all, 'acc_r': self.acc_real, 'acc_f': self.acc_fake,
            'weiG': self.last_wei_g if (self.last_wei_g is None or math.isfinite(self.last_wei_g)) else -23333,
            'grad': self.grad_boom,
            
            'cur': self.cur_phase, 'cur_ep': self.cur_ep, 'cur_it': self.cur_it,
            'rema': self.remain_time, 'fini': self.finish_time, 'last_upd': time.strftime("%Y-%m-%d %H:%M", time.localtime()),
            'bsep': f'{self.glb_batch_size}/{self.ep}',
            'G_lrwd': f'{self.glr:.1e}'.replace('.0', '').replace('-0', '-').replace('+0', '+') + f'/{self.gwd:g}',
            'D_lrwd': f'{self.dlr:.1e}'.replace('.0', '').replace('-0', '-').replace('+0', '+') + f'/{self.dwd:g}',
            'T_lrwd': f'{self.tlr:.1e}'.replace('.0', '').replace('-0', '-').replace('+0', '+') + f'/{self.twd:g}',
            'diff': self.diff, 'diffs': self.diffs, 'diffs_ema': self.diffs_ema if self.diffs_ema else None,
            'opt': self.opt,
            'is_master_node': self.is_master_node,
        }.items():
            if hasattr(v, 'item'):v = v.item()
            if v is None or (isinstance(v, str) and len(v) == 0): continue
            nd[k] = v
        if r_trial == trial:
            nd.pop('trial', None)
        
        with open(self.log_txt_path, 'w') as fp:
            json.dump(nd, fp, indent=2)
    
    def touch_log(self):
        os.utime(self.log_txt_path)
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            if k in {'is_large_model', 'gpt_training'}:
                continue
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'
    
    # =============== Circuit-breaker 相关方法 ===============
    def to_dict(self):
        """Circuit-breaker 兼容方法"""
        return dict( 
            target_layers=self.target_layers, 
            transform_layers=self.transform_layers,
            lorra_alpha=self.lorra_alpha, 
            trainsets=self.trainsets,
            valsets=self.valsets,
            full_layers=self.full_layers
        )


def init_dist_and_get_combined_args():
    """初始化分布式环境并获取合并后的参数"""
    import sys
    # Remove the circular import - we don't need to import from ourselves
    # sys.path.append('/home/gs285/VAR/my_model/circuit-breaker_original')
    # from src.combined_args import _scan_bad_defaults
    
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    
    args = CombinedArgs(explicit_bool=True).parse_args(known_only=True)
    

    
    print("BAD CLASS DEFAULTS =>", CombinedArgs.scan_class_defaults())
    print("BAD INSTANCE VALUES =>", CombinedArgs._scan_bad_defaults(args))
    args.chunk_nodes = int(os.environ.get('CK', '') or '0')
    
    # Set dynamic values that couldn't be set at class definition time
    try:
        args.branch = subprocess.check_output(f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]'
    except:
        args.branch = '[unknown]'
    
    args.cmd = ' '.join(a.replace('--exp_name=', '').replace('--exp_name ', '') for a in sys.argv[7:]) if len(sys.argv) > 7 else ''
    
    if len(args.extra_args) > 0 and args.is_master_node == 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    args.set_tf32(args.tf32)
    if args.dbg:
        torch.autograd.set_detect_anomaly(True)
    
    try: os.makedirs(args.bed, exist_ok=True)
    except: pass
    try: os.makedirs(args.local_out_path, exist_ok=True)
    except: pass
    
    day3 = 60*24*3
    dist.init_distributed_mode(local_out_path=args.local_out_path, fork=False, timeout_minutes=day3 if int(os.environ.get('LONG_DBG', '0') or '0') > 0 else 30)
    
    args.tlen = max(args.tlen, args.nodata_tlen)
    if args.zero and args.tema != 0:
        args.tema = 0
        print(f'======================================================================================')
        print(f'======================== WARNING: args.tema:=0, due to zero={args.zero} ========================')
        print(f'======================================================================================\n\n')
    
    if args.nodata:
        args.nova = True
    
    if not args.tos_profiler_file_prefix.endswith('/'): args.tos_profiler_file_prefix += '/'
    
    if args.alng < 0:
        args.alng = args.aln
    
    args.device = dist.get_device()
    args.r_accu = 1 / args.ac
    args.data_load_reso = None
    args.rand |= args.seed is None
    args.sche = args.sche or ('lin0' if args.gpt_training else 'cos')
    if args.wp == 0:
        args.wp = args.ep * 1/100
    
    di = {
        'b': 'bilinear', 'c': 'bicubic', 'n': 'nearest', 'a': 'area', 'aa': 'area+area',
        'at': 'auto', 'auto': 'auto',
        'v': 'vae',
        'x': 'pix', 'xg': 'pix_glu', 'gx': 'pix_glu', 'g': 'pix_glu'
    }
    
    args.ada = args.ada or ('0.9_0.96' if args.gpt_training else '0.5_0.9')
    args.dada = args.dada or args.ada
    args.opt = args.opt.lower().strip()
    
    if args.lbs:
        bs_per_gpu = args.lbs / args.ac
    else:
        bs_per_gpu = args.bs / args.ac / dist.get_world_size()
    bs_per_gpu = round(bs_per_gpu)
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size()
    args.workers = min(args.workers, bs_per_gpu)
    args.dblr = args.dblr or args.gblr
    args.glr = args.ac * args.gblr * args.glb_batch_size / 256
    args.dlr = args.ac * args.dblr * args.glb_batch_size / 256
    args.tlr = args.ac * args.tblr * args.glb_batch_size / 256
    args.gwde = args.gwde or args.gwd
    args.dwde = args.dwde or args.dwd
    args.twde = args.twde or args.twd
    
    if args.dbg_modified:
        torch.autograd.set_detect_anomaly(True)
    args.dbg_ks &= dist.is_local_master()
    if args.dbg_ks:
        args.dbg_ks_fp = open(os.path.join(args.local_out_path, 'dbg_ks.txt'), 'w')
    
    # gpt args
    if args.gpt_training:
        assert args.vae_ckpt, 'VAE ckpt must be specified when training GPT'
        from infinity.models import alias_dict, alias_dict_inv
        if args.model in alias_dict:
            args.model = alias_dict[args.model]
            args.model_alias = alias_dict_inv[args.model]
        else:
            args.model_alias = args.model
            args.model = f'infinity_{args.model}'
    
    args.task_id = '123'
    args.trial_id = '123'
    args.robust_run_id = '0'
    args.log_txt_path = os.path.join(args.local_out_path, 'log.txt')
    
    ls = []
    if 'AUTO_RESUME' in os.environ:
        ls.append(int(os.environ['AUTO_RESUME']))
    ls = sorted(ls, reverse=True)
    ls = [str(i) for i in ls]
    args.ckpt_trials = ls
    args.real_trial_id = args.trial_id if len(ls) == 0 else str(ls[-1])
    
    args.enable_checkpointing = None if args.enable_checkpointing in [False, 0, "0"] else args.enable_checkpointing
    args.enable_checkpointing = "full-block" if args.enable_checkpointing in [True, 1, "1"] else args.enable_checkpointing
    assert args.enable_checkpointing in [None, "full-block", "full-attn", "self-attn"], \
        f"only support no-checkpointing or full-block/full-attn checkpointing, but got {args.enable_checkpointing}."
    
    if len(args.exp_name) == 0:
        args.exp_name = os.path.basename(args.bed) or 'test_exp'
    
    if '-' in args.exp_name:
        args.tag, args.exp_name = args.exp_name.split('-', maxsplit=1)
    else:
        args.tag = 'UK'
    
    if dist.is_master():
        os.system(f'rm -rf {os.path.join(args.bed, "ready-node*")} {os.path.join(args.local_out_path, "ready-node*")}')
    
    if args.sdpa_mem:
        from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(False)
    
    return args 