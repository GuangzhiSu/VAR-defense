export CUDA_VISIBLE_DEVICES=3
nohup python3 /home/gs285/VAR/my_model/prompt_generation/prompt_generation.py \
    --prompt-model-name cognitivecomputations/Wizard-Vicuna-13B-Uncensored \
    --category hate \
    --num-prompts 200 \
    --output_dir /home/gs285/VAR/my_model/prompt_generation/output \
    --gpu_device 0 \
    > /home/gs285/VAR/my_model/prompt_generation/output/prompt_generation_hate.log 2>&1 &