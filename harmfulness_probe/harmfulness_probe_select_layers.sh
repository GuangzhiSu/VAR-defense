export CUDA_VISIBLE_DEVICES=3
nohup python3 /home/gs285/VAR/my_model/harmfulness_probe/harmfulness_probe_select_layer_sweep.py > /home/gs285/VAR/my_model/harmfulness_probe/log/harmfulness_probe_select_layers.log 2>&1 &
