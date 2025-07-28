export CUDA_VISIBLE_DEVICES=6

nohup python3 /home/gs285/VAR/my_model/harmfulness_probe/harmfulness_probe.py \
    --model_path /home/gs285/VAR/my_model/weights/infinity_8b_weights \
    --vae_path /home/gs285/VAR/my_model/weights/infinity_vae_d56_f8_14_patchify.pth \
    --text_encoder_ckpt google/flan-t5-xl \
    --output /home/gs285/VAR/my_model/harmfulness_probe/harmfulness_probe_results.json \
    --gpu_devices "0" \
    --cfg_insertion_layer 0 \
    --vae_type 14 \
    --sampling_per_bits 1 \
    --add_lvl_embeding_only_first_block 1 \
    --use_bit_label 1 \
    --model_type infinity_8b \
    > /home/gs285/VAR/my_model/harmfulness_probe/harmfulness_probe.log 2>&1 &