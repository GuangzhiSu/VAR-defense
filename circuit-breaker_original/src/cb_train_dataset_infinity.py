import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, T5TokenizerFast, T5EncoderModel
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
import torch.distributed as tdist
from torchvision.transforms.functional import to_tensor
from PIL import Image as PImage

class InfinityCircuitBreakerDataset(IterableDataset):
    """
    Dataset for Infinity model circuit breaker training
    Compatible with the training loop in train.py
    """
    
    def __init__(self, args, data_path, data_load_reso, max_caption_len=512, 
                 short_prob=0.2, load_vae_instead_of_image=False, buffersize=10000,
                 seed=0, pn='', online_t5=True, batch_size=2, num_replicas=1, 
                 rank=0, dataloader_workers=2, dynamic_resolution_across_gpus=True,
                 enable_dynamic_length_prompt=True, harmful_prompts_path=None, 
                 sanitized_prompts_path=None, validation_ratio=0.1, category=None,
                 **kwargs):
        
        # Store args for compatibility
        self.args = args
        self.data_path = data_path
        self.data_load_reso = data_load_reso
        self.max_caption_len = max_caption_len
        self.short_prob = short_prob
        self.load_vae_instead_of_image = load_vae_instead_of_image
        self.buffer_size = buffersize
        self.seed = seed
        self.pn = pn
        self.online_t5 = online_t5
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.dataloader_workers = max(1, dataloader_workers)
        self.dynamic_resolution_across_gpus = dynamic_resolution_across_gpus
        self.enable_dynamic_length_prompt = enable_dynamic_length_prompt
        
        # Circuit breaker specific parameters
        self.harmful_prompts_path = harmful_prompts_path
        self.sanitized_prompts_path = sanitized_prompts_path
        self.validation_ratio = validation_ratio
        self.category = category
        
        # Initialize worker info
        self.worker_id = 0
        self.global_worker_id = 0
        self.global_workers = self.num_replicas * self.dataloader_workers
        
        # Load data sources
        self.retain_data, self.validation_data = self._load_retain_and_validation_data()
        self.circuit_breaker_data = self._load_circuit_breaker_data()
        
        # Initialize text encoder and tokenizer
        self.text_encoder = self._init_text_encoder()
        self.tokenizer = self._init_tokenizer()
        
        # Initialize generators
        self.epoch_worker_generator = None
        self.epoch_global_worker_generator = None
        self.set_epoch(0)
        
        # Add missing attributes that Infinity training expects
        self.h_div_w_template2generator = {
            '1:1': self._create_generator('1:1'),
            '4:3': self._create_generator('4:3'),
            '3:4': self._create_generator('3:4'),
            '16:9': self._create_generator('16:9'),
            '9:16': self._create_generator('9:16'),
        }
        
        # Add other required attributes
        self.h_div_w_list = ['1:1', '4:3', '3:4', '16:9', '9:16']
        self.resolution_list = [512, 768, 1024]
        self.scale_list = [1, 2, 4, 8, 16, 32]
        
        print(f"Loaded {len(self.retain_data)} retain examples")
        print(f"Loaded {len(self.circuit_breaker_data)} circuit breaker examples")
        print(f"Loaded {len(self.validation_data)} validation examples")
        print(f'num_replicas: {num_replicas}, rank: {rank}, dataloader_workers: {dataloader_workers}, seed:{seed}')
    
    def _init_text_encoder(self):
        """Initialize T5 text encoder"""
        try:
            text_encoder = T5EncoderModel.from_pretrained('t5-base')
            return text_encoder
        except Exception as e:
            print(f"Warning: Could not load T5 encoder: {e}")
            return None
    
    def _init_tokenizer(self):
        """Initialize T5 tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained('t5-base')
            return tokenizer
        except Exception as e:
            print(f"Warning: Could not load T5 tokenizer: {e}")
            return None
    
    def _load_retain_and_validation_data(self):
        """Load sanitized prompts for retain training and validation"""
        retain_data = []
        validation_data = []
        
        # 构建sanitized prompts文件路径
        try:
            sanitized_prompts_path = self.sanitized_prompts_path
        except Exception as e:
            print("sanitized_prompts_path not found")
            return [], []
        
        # 尝试多个可能的文件名
        possible_files = [
            f"{self.category}_sanitized_prompts.json",
            "retain_prompts.json",
            "sanitized_prompts.json"
        ]
        
        sanitized_file_path = None
        for filename in possible_files:
            file_path = os.path.join(sanitized_prompts_path, filename)
            if os.path.exists(file_path):
                sanitized_file_path = file_path
                print(f"Found sanitized prompts file: {sanitized_file_path}")
                break
        
        if sanitized_file_path is None:
            print(f"Error: No sanitized prompts file found in {sanitized_prompts_path}")
            print(f"Tried: {possible_files}")
            return [], []
        
        # 加载sanitized prompts数据
        print(f"Loading sanitized prompts from {sanitized_file_path}")
        try:
            with open(sanitized_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    all_prompts = []
                    for item in data:
                        if isinstance(item, dict) and 'prompt' in item:
                            # 如果数据有category字段，只加载匹配category的数据
                            if 'category' in item and item['category'] == self.category:
                                all_prompts.append(item['prompt'])
                            elif 'category' not in item:
                                all_prompts.append(item['prompt'])
                        elif isinstance(item, str):
                            all_prompts.append(item)
                elif isinstance(data, dict) and 'prompts' in data:
                    all_prompts = []
                    for prompt_item in data['prompts']:
                        if isinstance(prompt_item, dict):
                            # Check for sanitized_prompt first, then prompt, then original_prompt
                            if 'sanitized_prompt' in prompt_item:
                                if 'category' in prompt_item and prompt_item['category'] == self.category:
                                    all_prompts.append(prompt_item['sanitized_prompt'])
                                elif 'category' not in prompt_item:
                                    all_prompts.append(prompt_item['sanitized_prompt'])
                            elif 'prompt' in prompt_item:
                                if 'category' in prompt_item and prompt_item['category'] == self.category:
                                    all_prompts.append(prompt_item['prompt'])
                                elif 'category' not in prompt_item:
                                    all_prompts.append(prompt_item['prompt'])
                            elif 'original_prompt' in prompt_item:
                                if 'category' in prompt_item and prompt_item['category'] == self.category:
                                    all_prompts.append(prompt_item['original_prompt'])
                                elif 'category' not in prompt_item:
                                    all_prompts.append(prompt_item['original_prompt'])
                        elif isinstance(prompt_item, str):
                            all_prompts.append(prompt_item)
                else:
                    print(f"Error: Unexpected data format in {sanitized_file_path}")
                    return [], []
                
                # 随机打乱数据
                random.shuffle(all_prompts)
                
                # 计算validation和retain的分割点
                validation_size = int(len(all_prompts) * self.validation_ratio)
                
                # 分割数据
                validation_prompts = all_prompts[:validation_size]
                retain_prompts = all_prompts[validation_size:]
                
                # 转换为数据格式
                for prompt in retain_prompts:
                    retain_data.append({
                        'prompt': prompt,
                        'type': 'retain'
                    })
                
                for prompt in validation_prompts:
                    validation_data.append({
                        'prompt': prompt,
                        'type': 'validation'
                    })
                
                print(f"Split {len(all_prompts)} prompts: {len(retain_prompts)} retain, {len(validation_prompts)} validation")
                
        except Exception as e:
            print(f"Error loading sanitized prompts: {e}")
            return [], []
        
        return retain_data, validation_data
    
    def _load_circuit_breaker_data(self):
        """Load harmful prompts for circuit breaker training"""
        circuit_breaker_data = []
        
        # 构建harmful prompts文件路径
        try:
            harmful_prompts_path = self.harmful_prompts_path
        except Exception as e:
            print("harmful_prompts_path not found")
            return []
        
        # 尝试多个可能的文件名
        possible_files = [
            f"{self.category}_prompts.json",
            "circuit_breaker_prompts.json",
            "harmful_prompts.json"
        ]
        
        harmful_file_path = None
        for filename in possible_files:
            file_path = os.path.join(harmful_prompts_path, filename)
            if os.path.exists(file_path):
                harmful_file_path = file_path
                print(f"Found harmful prompts file: {harmful_file_path}")
                break
        
        if harmful_file_path is None:
            print(f"Error: No harmful prompts file found in {harmful_prompts_path}")
            print(f"Tried: {possible_files}")
            return []
        
        # 加载harmful prompts数据
        print(f"Loading harmful prompts from {harmful_file_path}")
        try:
            with open(harmful_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'prompt' in item:
                            # 如果数据有category字段，只加载匹配category的数据
                            if 'category' in item and item['category'] == self.category:
                                circuit_breaker_data.append({
                                    'prompt': item['prompt'],
                                    'category': item.get('category', self.category or 'unknown'),
                                    'id': item.get('id', len(circuit_breaker_data) + 1),
                                    'type': 'circuit_breaker'
                                })
                            elif 'category' not in item:
                                circuit_breaker_data.append({
                                    'prompt': item['prompt'],
                                    'category': item.get('category', self.category or 'unknown'),
                                    'id': item.get('id', len(circuit_breaker_data) + 1),
                                    'type': 'circuit_breaker'
                                })
                        elif isinstance(item, str):
                            circuit_breaker_data.append({
                                'prompt': item,
                                'category': self.category or 'unknown',
                                'id': len(circuit_breaker_data) + 1,
                                'type': 'circuit_breaker'
                            })
                elif isinstance(data, dict) and 'prompts' in data:
                    category = data.get('category', self.category or 'unknown')
                    for prompt_item in data['prompts']:
                        if isinstance(prompt_item, dict) and 'prompt' in prompt_item:
                            circuit_breaker_data.append({
                                'prompt': prompt_item['prompt'],
                                'category': category,
                                'id': prompt_item.get('id', len(circuit_breaker_data) + 1),
                                'type': 'circuit_breaker'
                            })
                        elif isinstance(prompt_item, str):
                            circuit_breaker_data.append({
                                'prompt': prompt_item,
                                'category': category,
                                'id': len(circuit_breaker_data) + 1,
                                'type': 'circuit_breaker'
                            })
                else:
                    print(f"Error: Unexpected data format in {harmful_file_path}")
                    return []
                
        except Exception as e:
            print(f"Error loading harmful prompts: {e}")
            return []
        
        return circuit_breaker_data
    
    def set_global_worker_id(self):
        """Set global worker ID for distributed training"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
        else:
            worker_id = 0
            worker_total_num = 1
        
        # More flexible worker count validation - allow for slight mismatches
        if worker_total_num != self.dataloader_workers:
            print(f"Warning: Worker count mismatch. Expected {self.dataloader_workers}, got {worker_total_num}. Adjusting...")
            # Use the actual worker count from PyTorch
            self.dataloader_workers = worker_total_num
        
        self.worker_id = worker_id
        self.global_worker_id = self.rank * self.dataloader_workers + worker_id
    
    def set_epoch(self, epoch):
        """Set epoch for reproducible training"""
        self.epoch = epoch
        self.set_generator()
    
    def set_generator(self):
        """Set random generators for this epoch"""
        self.epoch_worker_generator = np.random.default_rng(self.seed + self.epoch + self.worker_id)
        self.epoch_global_worker_generator = np.random.default_rng(self.seed + self.epoch + self.global_worker_id)
    
    def _create_dummy_image(self, prompt):
        """Create a dummy image tensor for circuit breaker training (no actual images)"""
        # Create a dummy image tensor with shape (3, 512, 512) normalized to [-1, 1]
        # This is just a placeholder since circuit breaker training doesn't need real images
        dummy_image = torch.randn(3, 512, 512) * 0.1  # Small random values
        return dummy_image
    
    def _prepare_text_condition(self, text_prompt):
        """Prepare text condition tuple for Infinity model"""
        if self.text_encoder is None:
            # Fallback: create dummy text features
            dummy_features = torch.randn(1, 512, 768)  # (batch, seq_len, hidden_dim)
            lens = [512]
            cu_seqlens_k = torch.tensor([0, 512], dtype=torch.int32)
            Ltext = 512
            return (dummy_features, lens, cu_seqlens_k, Ltext)
        
        # Tokenize text
        if hasattr(self, 'tokenizer') and self.tokenizer:
            tokens = self.tokenizer(
                text_prompt, 
                max_length=512, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
        else:
            # Fallback tokenization
            input_ids = torch.randint(0, 1000, (1, 512))
            attention_mask = torch.ones(1, 512)
        
        # Encode with T5
        text_features = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )['last_hidden_state'].float()
        
        # Prepare condition tuple
        lens = attention_mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(attention_mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        Ltext = max(lens)
        
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)
        
        return (kv_compact, lens, cu_seqlens_k, Ltext)
    
    def __len__(self):
        """Return the number of batches"""
        # Return a reasonable number of batches for training
        min_data_length = min(len(self.retain_data), len(self.circuit_breaker_data), len(self.validation_data))
        if min_data_length == 0:
            return 100  # Default number of batches if no data
        return min_data_length
    
    def total_samples(self):
        """Return total number of samples"""
        return len(self) * self.batch_size * self.num_replicas
    
    def __iter__(self):
        """Iterate over batches of data"""
        self.set_global_worker_id()
        self.set_generator()
        
        # Calculate number of batches
        num_batches = len(self)
        
        for batch_idx in range(num_batches):
            batch_data = []
            
            # Create a batch of data
            for _ in range(self.batch_size):
                # Get data from different sources - handle empty data gracefully
                if len(self.retain_data) > 0:
                    retain_idx = (batch_idx * self.batch_size + _) % len(self.retain_data)
                    retain_data = self.retain_data[retain_idx]
                    prompt = retain_data['prompt']
                else:
                    # Use circuit breaker data if retain data is empty
                    prompt = "Default training prompt for circuit breaker"
                
                # Create dummy image for the prompt
                dummy_image = self._create_dummy_image(prompt)
                batch_data.append((prompt, dummy_image))
            
            # Prepare batch output in the format expected by train.py
            captions = [item[0] for item in batch_data]
            images = torch.stack([item[1] for item in batch_data])
            
            yield (images, captions)
    
    def getitem(self, i) -> Dict[str, torch.Tensor]:
        """Get a training example in the format needed for circuit breaker training"""
        
        # 获取数据 - handle empty data gracefully
        if len(self.retain_data) > 0:
            retain_data = self.retain_data[i % len(self.retain_data)]
            retain_prompt = retain_data['prompt']
        else:
            retain_prompt = "Default training prompt for circuit breaker"
        
        if len(self.circuit_breaker_data) > 0:
            circuit_breaker_data = self.circuit_breaker_data[i % len(self.circuit_breaker_data)]
            cb_prompt = circuit_breaker_data['prompt']
        else:
            cb_prompt = "Default circuit breaker prompt"
        
        if len(self.validation_data) > 0:
            validation_data = self.validation_data[i % len(self.validation_data)]
            val_prompt = validation_data['prompt']
        else:
            val_prompt = "Default validation prompt"
        
        # 准备文本条件 (用于Infinity模型)
        retain_text_cond = self._prepare_text_condition(retain_prompt)
        cb_text_cond = self._prepare_text_condition(cb_prompt)
        val_text_cond = self._prepare_text_condition(val_prompt)
        
        # 准备tokenized输入 (用于circuit breaker loss计算)
        retain_input_ids, retain_attention_mask = self._tokenize_prompt(retain_prompt)
        cb_input_ids, cb_attention_mask = self._tokenize_prompt(cb_prompt)
        val_input_ids, val_attention_mask = self._tokenize_prompt(val_prompt)
        
        return {
            # Infinity模型需要的文本条件
            'text_cond_tuple': retain_text_cond,
            'text_cond_tuple_circuit_breaker': cb_text_cond,
            'text_cond_tuple_val': val_text_cond,
            
            # Circuit breaker loss计算需要的tokenized输入
            'input_ids': retain_input_ids,
            'attention_mask': retain_attention_mask,
            'input_ids_circuit_breaker': cb_input_ids,
            'attention_mask_circuit_breaker': cb_attention_mask,
            'input_ids_val': val_input_ids,
            'attention_mask_val': val_attention_mask,
            
            # 原始prompt (用于调试)
            'retain_prompt': retain_prompt,
            'cb_prompt': cb_prompt,
            'val_prompt': val_prompt,
        }
    
    def _tokenize_prompt(self, prompt):
        """Tokenize prompt for circuit breaker training"""
        if hasattr(self, 'tokenizer') and self.tokenizer:
            tokens = self.tokenizer(
                prompt,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)
        else:
            # Fallback tokenization
            input_ids = torch.randint(0, 1000, (512,))
            attention_mask = torch.ones(512)
            return input_ids, attention_mask
    
    def _create_generator(self, h_div_w):
        """Create a generator for a specific aspect ratio"""
        def generator():
            while True:
                # Return dummy data for the aspect ratio
                yield {
                    'h_div_w': h_div_w,
                    'resolution': 512,
                    'scale': 1
                }
        return generator
    
    def get_resolution_and_scale(self, h_div_w):
        """Get resolution and scale for a given aspect ratio"""
        # Return default values
        return 512, 1
    
    def get_h_div_w_list(self):
        """Get list of available aspect ratios"""
        return self.h_div_w_list
    
    def get_resolution_list(self):
        """Get list of available resolutions"""
        return self.resolution_list
    
    def get_scale_list(self):
        """Get list of available scales"""
        return self.scale_list 

def build_circuit_breaker_dataset(
    args, 
    data_path, 
    data_load_reso, 
    max_caption_len=512, 
    short_prob=0.2, 
    load_vae_instead_of_image=False,
    harmful_prompts_path=None,
    sanitized_prompts_path=None,
    validation_ratio=0.1,
    category=None,
    **kwargs
):
    """
    Build circuit breaker dataset for Infinity model training
    This function has the same interface as build_t2i_dataset
    """
    return InfinityCircuitBreakerDataset(
        args=args,
        data_path=data_path,
        data_load_reso=data_load_reso,
        max_caption_len=max_caption_len,
        short_prob=short_prob,
        load_vae_instead_of_image=load_vae_instead_of_image,
        harmful_prompts_path=harmful_prompts_path,
        sanitized_prompts_path=sanitized_prompts_path,
        validation_ratio=validation_ratio,
        category=category,
        **kwargs
    ) 

# Example usage:
# In your training script, replace:
# from infinity.dataset.build import build_t2i_dataset
# 
# With:
# from cb_train_dataset_infinity import build_circuit_breaker_dataset
# 
# And replace:
# dataset_train = build_t2i_dataset(
#     args, 
#     args.data_path, 
#     args.data_load_reso, 
#     max_caption_len=args.tlen, 
#     short_prob=args.short_cap_prob, 
#     load_vae_instead_of_image=False
# )
# 
# With:
# dataset_train = build_circuit_breaker_dataset(
#     args, 
#     args.data_path, 
#     args.data_load_reso, 
#     max_caption_len=args.tlen, 
#     short_prob=args.short_cap_prob, 
#     load_vae_instead_of_image=False,
#     harmful_prompts_path=args.harmful_prompts_path,
#     sanitized_prompts_path=args.sanitized_prompts_path,
#     validation_ratio=args.validation_ratio,
#     category=args.category
# ) 