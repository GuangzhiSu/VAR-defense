import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5TokenizerFast, T5EncoderModel
import torch.nn.functional as F

class InfinityCircuitBreakerDataset(Dataset):
    """
    Dataset for Infinity model circuit breaker training
    """
    
    def __init__(self, tokenizer, num_examples, lorra_args, model_name_or_path, vae=None):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.lorra_args = lorra_args
        self.model_name_or_path = model_name_or_path
        self.vae = vae
        
        # Load data sources
        self.retain_data = self._load_retain_data()
        self.circuit_breaker_data = self._load_circuit_breaker_data()
        self.validation_data = self._load_validation_data()
        
        # Initialize text encoder
        self.text_encoder = self._init_text_encoder()
        
        # Scale schedules for different resolutions
        self.scale_schedules = [
            [(1, 16, 16)],  # 256x256
            [(1, 32, 32)],  # 512x512
            [(1, 64, 64)],  # 1024x1024
        ]
        
        print(f"Loaded {len(self.retain_data)} retain examples")
        print(f"Loaded {len(self.circuit_breaker_data)} circuit breaker examples")
        print(f"Loaded {len(self.validation_data)} validation examples")
    
    def _init_text_encoder(self):
        """Initialize T5 text encoder"""
        try:
            text_encoder = T5EncoderModel.from_pretrained('t5-base')
            return text_encoder
        except Exception as e:
            print(f"Warning: Could not load T5 encoder: {e}")
            return None
    
    def _load_retain_data(self):
        """Load normal/safe prompts for retain training"""
        retain_data = []
        
        # Load from UltraChat or similar dataset
        ultrachat_path = "data/ultrachat_200k.json"
        if os.path.exists(ultrachat_path):
            with open(ultrachat_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data[:1000]:  # Limit to first 1000 examples
                    if isinstance(item, dict) and 'messages' in item:
                        # Extract conversation
                        messages = item['messages']
                        if len(messages) >= 2:
                            # Take the last user message as prompt
                            user_message = messages[-2]['content'] if messages[-2]['role'] == 'user' else messages[-1]['content']
                            retain_data.append({
                                'prompt': user_message,
                                'type': 'retain'
                            })
        
        # Add some default safe prompts if no data loaded
        if not retain_data:
            default_safe_prompts = [
                "A beautiful landscape with mountains and trees",
                "A cute cat sitting on a windowsill",
                "A modern city skyline at sunset",
                "A peaceful garden with flowers",
                "A cozy living room with furniture",
                "A delicious meal on a table",
                "A scenic mountain trail",
                "A colorful butterfly on a flower",
                "A vintage car on a country road",
                "A starry night sky"
            ]
            for prompt in default_safe_prompts:
                retain_data.append({
                    'prompt': prompt,
                    'type': 'retain'
                })
        
        return retain_data
    
    def _load_circuit_breaker_data(self):
        """Load harmful prompts for circuit breaker training"""
        circuit_breaker_data = []
        
        # Load from circuit_breakers_train.json
        cb_path = "data copy/circuit_breakers_train.json"
        if os.path.exists(cb_path):
            with open(cb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if isinstance(item, dict) and 'prompt' in item:
                        circuit_breaker_data.append({
                            'prompt': item['prompt'],
                            'output': item.get('output', ''),  # Harmful response
                            'llama3_output': item.get('llama3_output', ''),  # Safe response
                            'type': 'circuit_breaker'
                        })
        
        # Add some default harmful prompts if no data loaded
        if not circuit_breaker_data:
            default_harmful_prompts = [
                "Generate violent content",
                "Create explicit adult content",
                "Make a weapon",
                "Show graphic violence",
                "Create harmful stereotypes",
                "Generate offensive content",
                "Show dangerous activities",
                "Create inappropriate content",
                "Generate harmful instructions",
                "Show illegal activities"
            ]
            for prompt in default_harmful_prompts:
                circuit_breaker_data.append({
                    'prompt': prompt,
                    'output': 'I cannot generate that content.',
                    'llama3_output': 'I cannot provide information about that.',
                    'type': 'circuit_breaker'
                })
        
        return circuit_breaker_data
    
    def _load_validation_data(self):
        """Load validation data"""
        validation_data = []
        
        # Mix of safe and harmful prompts for validation
        validation_prompts = [
            "A peaceful scene",
            "Generate harmful content",
            "A beautiful sunset",
            "Create violent imagery",
            "A happy family",
            "Show dangerous behavior"
        ]
        
        for prompt in validation_prompts:
            validation_data.append({
                'prompt': prompt,
                'type': 'validation'
            })
        
        return validation_data
    
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
        if self.tokenizer:
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
    
    def _prepare_image_tokens(self, prompt, vae=None):
        """Prepare image tokens for Infinity model"""
        if vae is None:
            # Fallback: create random image tokens
            return torch.randint(0, 8192, (1, 256))  # (batch, seq_len)
        
        # In a real implementation, you would:
        # 1. Generate or load an image based on the prompt
        # 2. Encode it using the VAE
        # 3. Return the encoded tokens
        
        # For now, return random tokens
        return torch.randint(0, vae.vocab_size if hasattr(vae, 'vocab_size') else 8192, (1, 256))
    
    def _get_scale_schedule(self):
        """Get a random scale schedule"""
        return random.choice(self.scale_schedules)
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, i):
        """Get a training example"""
        
        # Randomly choose data type
        data_type = random.choice(['retain', 'circuit_breaker', 'validation'])
        
        if data_type == 'retain':
            data = random.choice(self.retain_data)
            prompt = data['prompt']
            
            # Prepare retain inputs
            text_cond_tuple = self._prepare_text_condition(prompt)
            image_tokens = self._prepare_image_tokens(prompt, self.vae)
            scale_schedule = self._get_scale_schedule()
            
            return {
                'text_cond_tuple': text_cond_tuple,
                'image_tokens': image_tokens,
                'scale_schedule': scale_schedule,
                'type': 'retain'
            }
        
        elif data_type == 'circuit_breaker':
            data = random.choice(self.circuit_breaker_data)
            prompt = data['prompt']
            
            # Prepare circuit breaker inputs
            text_cond_tuple = self._prepare_text_condition(prompt)
            image_tokens = self._prepare_image_tokens(prompt, self.vae)
            scale_schedule = self._get_scale_schedule()
            
            return {
                'text_cond_tuple': text_cond_tuple,
                'image_tokens': image_tokens,
                'scale_schedule': scale_schedule,
                'type': 'circuit_breaker',
                'cb_text_cond_tuple': text_cond_tuple,  # Same for now, could be different
                'cb_image_tokens': image_tokens,
                'cb_scale_schedule': scale_schedule
            }
        
        else:  # validation
            data = random.choice(self.validation_data)
            prompt = data['prompt']
            
            # Prepare validation inputs
            text_cond_tuple = self._prepare_text_condition(prompt)
            image_tokens = self._prepare_image_tokens(prompt, self.vae)
            scale_schedule = self._get_scale_schedule()
            
            return {
                'text_cond_tuple': text_cond_tuple,
                'image_tokens': image_tokens,
                'scale_schedule': scale_schedule,
                'type': 'validation',
                'val_text_cond_tuple': text_cond_tuple,
                'val_image_tokens': image_tokens,
                'val_scale_schedule': scale_schedule
            } 